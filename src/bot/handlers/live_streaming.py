"""Live streaming handler for Claude Code with separate Telegram messages.

This module handles real-time Claude updates as separate Telegram messages,
showing tools being used, todos, and progress with a cancel button.
"""

import asyncio
import re
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
    tool_messages: Dict[str, Message] = field(default_factory=dict)
    todo_message: Optional[Message] = None
    current_todos: List[Dict] = field(default_factory=list)

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
        """Handle assistant message - check for todos and send as separate message."""
        content = update.content

        # Extract todos from message
        todos = self._extract_todos(content)
        if todos and todos != context.current_todos:
            context.current_todos = todos
            await self._update_todo_display(context, todos)

        # Update status message
        preview = content[:100] + "..." if len(content) > 100 else content
        await self._update_status(context, f"💬 **Claude is responding...**\n\n_{preview}_")

    async def _handle_tool_calls(
        self,
        context: LiveStreamContext,
        update: StreamUpdate
    ):
        """Handle tool calls - send separate message for each tool."""
        if not update.tool_calls:
            return

        for tool_call in update.tool_calls:
            tool_name = tool_call.get("name", "Unknown")
            tool_id = tool_call.get("id", str(context.tools_count))

            # Create tool message
            tool_emoji = self._get_tool_emoji(tool_name)
            tool_text = f"{tool_emoji} **Using tool: {tool_name}**\n\n_Executing..._"

            keyboard = [[
                InlineKeyboardButton("🛑 Stop", callback_data=f"cancel:{context.process_id}")
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            tool_msg = await self.bot.send_message(
                chat_id=context.chat_id,
                text=tool_text,
                parse_mode="Markdown",
                reply_markup=reply_markup
            )

            context.tool_messages[tool_id] = tool_msg
            context.tools_count += 1
            context.messages_sent += 1

    async def _handle_tool_result(
        self,
        context: LiveStreamContext,
        update: StreamUpdate
    ):
        """Handle tool result - update the corresponding tool message."""
        tool_id = update.metadata.get("tool_use_id") if update.metadata else None
        if not tool_id or tool_id not in context.tool_messages:
            return

        tool_msg = context.tool_messages[tool_id]
        tool_name = update.metadata.get("tool_name", "Tool") if update.metadata else "Tool"
        tool_emoji = self._get_tool_emoji(tool_name)

        if update.is_error():
            result_text = f"{tool_emoji} **{tool_name}**\n\n❌ Failed: {update.get_error_message()}"
        else:
            result_text = f"{tool_emoji} **{tool_name}**\n\n✅ Completed"

        try:
            await tool_msg.edit_text(
                result_text,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.warning("Failed to edit tool message", error=str(e))

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

        # Remove cancel buttons from tool messages
        for tool_msg in context.tool_messages.values():
            try:
                await tool_msg.edit_reply_markup(reply_markup=None)
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
            tools_used=context.tools_count
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
