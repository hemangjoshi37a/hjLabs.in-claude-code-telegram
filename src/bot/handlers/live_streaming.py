"""Live streaming handler for Claude Code with separate Telegram messages.

This module handles real-time Claude updates as separate Telegram messages,
showing tools being used, todos, and progress with a cancel button.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import ContextTypes

from ...claude.integration import StreamUpdate

logger = structlog.get_logger()


@dataclass
class LiveStreamContext:
    """Context for managing live stream messages."""

    user_id: int
    chat_id: int
    original_message_id: int

    # Message tracking
    status_message: Optional[Message] = None
    content_message: Optional[Message] = None  # Main response message
    tool_messages: Dict[str, Message] = field(default_factory=dict)
    todo_message: Optional[Message] = None
    current_todos: List[Dict] = field(default_factory=list)

    # Content streaming
    accumulated_content: str = ""
    last_update_time: float = 0.0
    update_throttle: float = 0.3  # Update every 300ms max

    # Control
    cancel_requested: bool = False
    process_id: Optional[str] = None

    # Stats
    tools_count: int = 0
    messages_sent: int = 0


class LiveStreamHandler:
    """Handle live streaming updates from Claude Code as separate Telegram messages."""

    def __init__(self, bot_application):
        self.bot = bot_application.bot
        self.active_streams: Dict[str, LiveStreamContext] = {}

    async def start_stream(
        self,
        user_id: int,
        chat_id: int,
        original_message_id: int,
        process_id: str
    ) -> LiveStreamContext:
        """Start a new live stream session."""
        context = LiveStreamContext(
            user_id=user_id,
            chat_id=chat_id,
            original_message_id=original_message_id,
            process_id=process_id
        )

        # Create initial status message with cancel button
        keyboard = [[InlineKeyboardButton("🛑 Stop Claude", callback_data=f"cancel:{process_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        status_msg = await self.bot.send_message(
            chat_id=chat_id,
            text="🤖 **Claude is starting...**",
            parse_mode="Markdown",
            reply_markup=reply_markup,
            reply_to_message_id=original_message_id
        )

        context.status_message = status_msg
        self.active_streams[process_id] = context

        logger.info(
            "Started live stream",
            user_id=user_id,
            process_id=process_id
        )

        return context

    async def handle_update(
        self,
        process_id: str,
        update: StreamUpdate
    ):
        """Handle a stream update and send appropriate Telegram messages."""
        context = self.active_streams.get(process_id)
        if not context:
            logger.warning("Stream context not found", process_id=process_id)
            return

        # Check for cancellation
        if context.cancel_requested:
            logger.info("Stream cancelled by user", process_id=process_id)
            return

        try:
            # Handle different update types
            if update.type == "assistant" and update.content:
                await self._handle_assistant_message(context, update)

            elif update.type == "assistant" and update.tool_calls:
                await self._handle_tool_calls(context, update)

            elif update.type == "tool_result":
                await self._handle_tool_result(context, update)

            elif update.type == "progress":
                await self._handle_progress(context, update)

            elif update.type == "error":
                await self._handle_error(context, update)

        except Exception as e:
            logger.error(
                "Error handling stream update",
                error=str(e),
                update_type=update.type,
                process_id=process_id
            )

    async def _handle_assistant_message(
        self,
        context: LiveStreamContext,
        update: StreamUpdate
    ):
        """Handle assistant message - stream content live to Telegram."""
        if not update.content:
            return

        # Accumulate content
        context.accumulated_content += update.content

        # Extract todos from accumulated message
        todos = self._extract_todos(context.accumulated_content)
        if todos and todos != context.current_todos:
            context.current_todos = todos
            await self._update_todo_display(context, todos)

        # Throttle updates to avoid hitting Telegram rate limits
        # Update every 300ms or when content is substantial
        current_time = time.time()
        time_since_update = current_time - context.last_update_time
        content_length = len(context.accumulated_content)

        should_update = (
            time_since_update >= context.update_throttle or
            content_length % 200 == 0  # Update every ~200 chars
        )

        if should_update:
            await self._update_content_message(context)
            context.last_update_time = current_time

    async def _update_content_message(
        self,
        context: LiveStreamContext
    ):
        """Update or create the content message with accumulated text."""
        if not context.accumulated_content:
            return

        # Prepare content with stop button
        keyboard = [[InlineKeyboardButton("🛑 Stop", callback_data=f"cancel:{context.process_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Truncate if too long (Telegram limit is 4096 chars)
        content = context.accumulated_content
        if len(content) > 4000:
            content = content[:4000] + "\n\n_...response continues..._"

        try:
            if context.content_message:
                # Update existing message
                await context.content_message.edit_text(
                    content,
                    parse_mode="Markdown",
                    reply_markup=reply_markup
                )
            else:
                # Create new content message
                context.content_message = await self.bot.send_message(
                    chat_id=context.chat_id,
                    text=content,
                    parse_mode="Markdown",
                    reply_markup=reply_markup
                )
                context.messages_sent += 1

                # Update status to show we're streaming
                await self._update_status(context, "💬 **Streaming response...**")

        except Exception as e:
            # Fallback to plain text if markdown fails
            try:
                if context.content_message:
                    await context.content_message.edit_text(content)
                else:
                    context.content_message = await self.bot.send_message(
                        chat_id=context.chat_id,
                        text=content
                    )
            except Exception as e2:
                logger.warning("Failed to update content message", error=str(e2))

    async def _handle_tool_calls(
        self,
        context: LiveStreamContext,
        update: StreamUpdate
    ):
        """Handle tool calls - update status message instead of creating new messages."""
        if not update.tool_calls:
            return

        # Collect all tool names
        tool_names = []
        for tool_call in update.tool_calls:
            tool_name = tool_call.get("name", "Unknown")
            tool_id = tool_call.get("id", str(context.tools_count))

            tool_emoji = self._get_tool_emoji(tool_name)
            tool_names.append(f"{tool_emoji} {tool_name}")

            # Track tool IDs for potential result updates
            context.tool_messages[tool_id] = None
            context.tools_count += 1

        # Update status message with current tools
        if tool_names:
            tools_text = ", ".join(tool_names)
            status_text = f"🔧 **Using tools:** {tools_text}\n\n_Working..._"
            await self._update_status(context, status_text)

    async def _handle_tool_result(
        self,
        context: LiveStreamContext,
        update: StreamUpdate
    ):
        """Handle tool result - just update status if there's an error."""
        if update.is_error():
            error_msg = update.get_error_message()
            await self._update_status(context, f"⚠️ **Tool error:** {error_msg}")

    async def _handle_progress(
        self,
        context: LiveStreamContext,
        update: StreamUpdate
    ):
        """Handle progress updates."""
        progress_text = update.content or "Working..."
        percentage = update.get_progress_percentage()

        status_text = f"🔄 **{progress_text}**"

        if percentage is not None:
            # Create progress bar
            filled = int(percentage / 10)
            bar = "█" * filled + "░" * (10 - filled)
            status_text += f"\n\n`{bar}` {percentage}%"

        await self._update_status(context, status_text)

    async def _handle_error(
        self,
        context: LiveStreamContext,
        update: StreamUpdate
    ):
        """Handle error messages."""
        error_msg = update.get_error_message() or "An error occurred"

        await self.bot.send_message(
            chat_id=context.chat_id,
            text=f"❌ **Error**\n\n{error_msg}",
            parse_mode="Markdown"
        )

        context.messages_sent += 1

    async def _update_status(
        self,
        context: LiveStreamContext,
        text: str
    ):
        """Update the status message."""
        if not context.status_message:
            return

        keyboard = [[
            InlineKeyboardButton("🛑 Stop Claude", callback_data=f"cancel:{context.process_id}")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await context.status_message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
        except Exception as e:
            logger.warning("Failed to update status message", error=str(e))

    async def _update_todo_display(
        self,
        context: LiveStreamContext,
        todos: List[Dict]
    ):
        """Update or create todo list message."""
        if not todos:
            return

        # Format todos
        todo_text = "📋 **Claude's Task List:**\n\n"

        for i, todo in enumerate(todos, 1):
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅"
            }.get(todo.get("status", "pending"), "📌")

            content = todo.get("content", "Unknown task")
            todo_text += f"{status_icon} {content}\n"

        keyboard = [[
            InlineKeyboardButton("🛑 Stop", callback_data=f"cancel:{context.process_id}")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        if context.todo_message:
            # Update existing message
            try:
                await context.todo_message.edit_text(
                    todo_text,
                    parse_mode="Markdown",
                    reply_markup=reply_markup
                )
            except Exception as e:
                logger.warning("Failed to update todo message", error=str(e))
        else:
            # Create new message
            context.todo_message = await self.bot.send_message(
                chat_id=context.chat_id,
                text=todo_text,
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
            context.messages_sent += 1

    def _extract_todos(self, content: str) -> List[Dict]:
        """Extract todo items from Claude's message.

        Looks for TodoWrite tool usage or formatted todo lists in the content.
        """
        todos = []

        # Pattern 1: Look for TodoWrite mentions
        # Pattern: "- [ ] task" or "- [x] task" or "1. task (status)"
        todo_patterns = [
            r'[-*]\s*\[[ x]\]\s*(.+?)(?:\n|$)',  # - [ ] or - [x] format
            r'\d+\.\s*(.+?)\s*\((\w+)\)',  # 1. task (pending) format
            r'(?:⏳|🔄|✅)\s*(.+?)(?:\n|$)',  # Emoji prefix format
        ]

        for pattern in todo_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    content_text = match[0].strip()
                    status = match[1] if len(match) > 1 else "pending"
                else:
                    content_text = match.strip()
                    status = "pending"

                # Determine status from content or markers
                if "✅" in content_text or "[x]" in content.lower():
                    status = "completed"
                elif "🔄" in content_text:
                    status = "in_progress"

                todos.append({
                    "content": content_text.replace("✅", "").replace("🔄", "").replace("⏳", "").strip(),
                    "status": status
                })

        return todos

    def _get_tool_emoji(self, tool_name: str) -> str:
        """Get emoji for tool name."""
        emojis = {
            "Read": "📖",
            "Write": "✍️",
            "Edit": "📝",
            "Bash": "⚡",
            "Glob": "🔍",
            "Grep": "🔎",
            "Task": "🎯",
            "WebFetch": "🌐",
            "WebSearch": "🔍",
        }
        return emojis.get(tool_name, "🔧")

    async def finalize_stream(
        self,
        process_id: str,
        final_message: str,
        is_error: bool = False
    ):
        """Finalize the stream and clean up."""
        context = self.active_streams.get(process_id)
        if not context:
            return

        # Do final update of content message
        if context.accumulated_content:
            await self._update_content_message(context)

        # Remove stop button from content message
        if context.content_message:
            try:
                await context.content_message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass

        # Update status message with final state
        if context.status_message:
            try:
                icon = "❌" if is_error else "✅"
                await context.status_message.edit_text(
                    f"{icon} **Claude finished**",
                    parse_mode="Markdown"
                )
            except Exception:
                pass

        # Remove cancel button from todo message
        if context.todo_message:
            try:
                await context.todo_message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass

        logger.info(
            "Finalized stream",
            process_id=process_id,
            messages_sent=context.messages_sent,
            tools_used=context.tools_count,
            content_length=len(context.accumulated_content)
        )

        # Clean up
        del self.active_streams[process_id]

    def request_cancel(self, process_id: str) -> bool:
        """Request cancellation of a stream."""
        context = self.active_streams.get(process_id)
        if not context:
            return False

        context.cancel_requested = True
        logger.info("Cancel requested", process_id=process_id)
        return True

    def is_cancelled(self, process_id: str) -> bool:
        """Check if stream has been cancelled."""
        context = self.active_streams.get(process_id)
        return context.cancel_requested if context else False
