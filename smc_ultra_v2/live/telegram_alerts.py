"""
Telegram Alerts Module
======================
Professional notifications for trade events and daily/weekly/monthly summaries.

All values in % for universal display regardless of account size.

Setup:
1. Create bot with @BotFather on Telegram
2. Get bot token
3. Start chat with bot, send any message
4. Get chat ID: https://api.telegram.org/bot<TOKEN>/getUpdates
5. Set env vars:
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo

log = logging.getLogger("telegram")

# Config
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
BOT_NAME = os.getenv('BOT_NAME', 'OB Scalper')
TIMEZONE = ZoneInfo('Europe/Berlin')  # German time for summaries


def is_enabled() -> bool:
    """Check if Telegram is configured."""
    return bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


def send_message(text: str, silent: bool = False) -> bool:
    """Send message to Telegram. Returns True on success."""
    if not is_enabled():
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_notification": silent,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.debug(f"Telegram sent: {text[:50]}...")
            return True
        else:
            log.warning(f"Telegram error: {resp.status_code}")
            return False
    except Exception as e:
        log.warning(f"Telegram failed: {e}")
        return False


# ============================================
# TRADE ALERTS
# ============================================

def send_trade_opened(
    symbol: str,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float,
    leverage: int,
    risk_pct: float,
    ob_strength: float = None,
    ob_age: int = None,
) -> bool:
    """
    Send notification when new trade is opened.

    Shows: Symbol, Direction, Entry, SL%, TP targets, Leverage, Risk
    """
    if not is_enabled():
        return False

    emoji = "🔴" if direction == 'short' else "🟢"
    dir_text = "SHORT" if direction == 'short' else "LONG"

    # Calculate SL distance %
    sl_dist = abs(entry_price - sl_price) / entry_price * 100

    # Calculate TP distances %
    tp1_dist = abs(tp1_price - entry_price) / entry_price * 100
    tp2_dist = abs(tp2_price - entry_price) / entry_price * 100

    # Build message
    lines = [
        f"{emoji} <b>NEW TRADE</b>",
        "",
        f"<b>{symbol}</b>",
        f"Direction: {dir_text}",
        f"Leverage: {leverage}x",
        "",
        f"Entry: {entry_price:.6f}",
        f"SL: {sl_price:.6f} ({sl_dist:.2f}%)",
        f"TP1: {tp1_price:.6f} ({tp1_dist:.2f}%)",
        f"TP2: {tp2_price:.6f} ({tp2_dist:.2f}%)",
        "",
        f"Risk: {risk_pct:.1f}%",
    ]

    if ob_strength:
        lines.append(f"OB Strength: {ob_strength:.2f}")
    if ob_age:
        lines.append(f"OB Age: {ob_age} candles")

    lines.append("")
    lines.append(f"<i>{BOT_NAME}</i>")

    return send_message("\n".join(lines))


def send_trade_closed(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    exit_reason: str,
    tp_hits: str,  # "0/2", "1/2", "2/2"
    duration_mins: int = None,
    r_multiple: float = None,
) -> bool:
    """
    Send notification when trade is closed.

    Shows: Result, PnL%, TP hits, Duration, R-multiple
    """
    if not is_enabled():
        return False

    is_win = pnl_pct > 0
    emoji = "✅" if is_win else "❌"
    result = "WIN" if is_win else "LOSS"
    dir_text = "SHORT" if direction == 'short' else "LONG"

    # Format PnL
    pnl_str = f"+{pnl_pct:.2f}%" if pnl_pct > 0 else f"{pnl_pct:.2f}%"

    # Format exit reason nicely
    exit_map = {
        'tp1': 'Take Profit 1',
        'tp2': 'Take Profit 2',
        'sl': 'Stop Loss',
        'be': 'Break Even',
        'be+': 'Break Even+',
        'manual': 'Manual Close',
    }
    exit_text = exit_map.get(exit_reason, exit_reason.upper())

    # Build message
    lines = [
        f"{emoji} <b>TRADE CLOSED: {result}</b>",
        "",
        f"<b>{symbol}</b> {dir_text}",
        "",
        f"PnL: <b>{pnl_str}</b>",
        f"TPs: {tp_hits}",
        f"Exit: {exit_text}",
    ]

    if duration_mins is not None:
        if duration_mins < 60:
            dur_str = f"{duration_mins}m"
        elif duration_mins < 1440:
            dur_str = f"{duration_mins // 60}h {duration_mins % 60}m"
        else:
            dur_str = f"{duration_mins // 1440}d {(duration_mins % 1440) // 60}h"
        lines.append(f"Duration: {dur_str}")

    if r_multiple is not None:
        r_str = f"+{r_multiple:.2f}R" if r_multiple > 0 else f"{r_multiple:.2f}R"
        lines.append(f"R-Multiple: {r_str}")

    lines.append("")
    lines.append(f"<i>{BOT_NAME}</i>")

    return send_message("\n".join(lines))


def send_tp1_hit(
    symbol: str,
    direction: str,
    entry_price: float,
    tp1_price: float,
    partial_pnl_pct: float,
) -> bool:
    """Send notification when TP1 is hit (partial close)."""
    if not is_enabled():
        return False

    emoji = "🎯"
    dir_text = "SHORT" if direction == 'short' else "LONG"
    pnl_str = f"+{partial_pnl_pct:.2f}%"

    lines = [
        f"{emoji} <b>TP1 HIT</b>",
        "",
        f"<b>{symbol}</b> {dir_text}",
        f"50% closed at {tp1_price:.6f}",
        f"Partial PnL: {pnl_str}",
        "",
        f"SL → Break Even",
        f"Remaining 50% running to TP2",
        "",
        f"<i>{BOT_NAME}</i>",
    ]

    return send_message("\n".join(lines))


# ============================================
# SUMMARY REPORTS
# ============================================

def send_daily_summary(
    trades_opened: int,
    trades_closed: int,
    wins: int,
    losses: int,
    total_pnl_pct: float,
    best_trade_pct: float = None,
    worst_trade_pct: float = None,
    current_equity_pct: float = None,  # % change from start
) -> bool:
    """Send daily summary at 00:00 German time."""
    if not is_enabled():
        return False

    now = datetime.now(TIMEZONE)
    date_str = (now - timedelta(days=1)).strftime("%d.%m.%Y")

    win_rate = (wins / trades_closed * 100) if trades_closed > 0 else 0
    emoji = "📈" if total_pnl_pct >= 0 else "📉"
    pnl_str = f"+{total_pnl_pct:.2f}%" if total_pnl_pct >= 0 else f"{total_pnl_pct:.2f}%"

    lines = [
        f"📊 <b>DAILY REPORT</b>",
        f"<i>{date_str}</i>",
        "",
        f"Trades Opened: {trades_opened}",
        f"Trades Closed: {trades_closed}",
        "",
        f"Wins: {wins}",
        f"Losses: {losses}",
        f"Win Rate: {win_rate:.1f}%",
        "",
        f"{emoji} Day PnL: <b>{pnl_str}</b>",
    ]

    if best_trade_pct is not None:
        lines.append(f"Best Trade: +{best_trade_pct:.2f}%")
    if worst_trade_pct is not None:
        lines.append(f"Worst Trade: {worst_trade_pct:.2f}%")
    if current_equity_pct is not None:
        eq_str = f"+{current_equity_pct:.2f}%" if current_equity_pct >= 0 else f"{current_equity_pct:.2f}%"
        lines.append(f"Equity: {eq_str}")

    lines.append("")
    lines.append(f"<i>{BOT_NAME}</i>")

    return send_message("\n".join(lines))


def send_weekly_summary(
    trades_opened: int,
    trades_closed: int,
    wins: int,
    losses: int,
    total_pnl_pct: float,
    avg_win_pct: float = None,
    avg_loss_pct: float = None,
    win_rate: float = None,
    equity_change_pct: float = None,
) -> bool:
    """Send weekly summary on Monday 00:00 German time."""
    if not is_enabled():
        return False

    now = datetime.now(TIMEZONE)
    week_start = (now - timedelta(days=7)).strftime("%d.%m")
    week_end = (now - timedelta(days=1)).strftime("%d.%m.%Y")

    if win_rate is None:
        win_rate = (wins / trades_closed * 100) if trades_closed > 0 else 0

    emoji = "📈" if total_pnl_pct >= 0 else "📉"
    pnl_str = f"+{total_pnl_pct:.2f}%" if total_pnl_pct >= 0 else f"{total_pnl_pct:.2f}%"

    lines = [
        f"📅 <b>WEEKLY REPORT</b>",
        f"<i>{week_start} - {week_end}</i>",
        "",
        f"Total Trades: {trades_closed}",
        f"Wins: {wins} | Losses: {losses}",
        f"Win Rate: <b>{win_rate:.1f}%</b>",
        "",
        f"{emoji} Week PnL: <b>{pnl_str}</b>",
    ]

    if avg_win_pct is not None:
        lines.append(f"Avg Win: +{avg_win_pct:.2f}%")
    if avg_loss_pct is not None:
        lines.append(f"Avg Loss: {avg_loss_pct:.2f}%")
    if equity_change_pct is not None:
        eq_str = f"+{equity_change_pct:.2f}%" if equity_change_pct >= 0 else f"{equity_change_pct:.2f}%"
        lines.append(f"Equity Change: {eq_str}")

    lines.append("")
    lines.append(f"<i>{BOT_NAME}</i>")

    return send_message("\n".join(lines))


def send_monthly_summary(
    trades_opened: int,
    trades_closed: int,
    wins: int,
    losses: int,
    total_pnl_pct: float,
    best_day_pct: float = None,
    worst_day_pct: float = None,
    max_drawdown_pct: float = None,
    equity_change_pct: float = None,
) -> bool:
    """Send monthly summary on 1st of month 00:00 German time."""
    if not is_enabled():
        return False

    now = datetime.now(TIMEZONE)
    # Previous month
    if now.month == 1:
        month_name = "December"
        year = now.year - 1
    else:
        month_name = (now - timedelta(days=1)).strftime("%B")
        year = now.year

    win_rate = (wins / trades_closed * 100) if trades_closed > 0 else 0
    emoji = "📈" if total_pnl_pct >= 0 else "📉"
    pnl_str = f"+{total_pnl_pct:.2f}%" if total_pnl_pct >= 0 else f"{total_pnl_pct:.2f}%"

    lines = [
        f"📆 <b>MONTHLY REPORT</b>",
        f"<i>{month_name} {year}</i>",
        "",
        f"Total Trades: {trades_closed}",
        f"Wins: {wins} | Losses: {losses}",
        f"Win Rate: <b>{win_rate:.1f}%</b>",
        "",
        f"{emoji} Month PnL: <b>{pnl_str}</b>",
    ]

    if best_day_pct is not None:
        lines.append(f"Best Day: +{best_day_pct:.2f}%")
    if worst_day_pct is not None:
        lines.append(f"Worst Day: {worst_day_pct:.2f}%")
    if max_drawdown_pct is not None:
        lines.append(f"Max Drawdown: {max_drawdown_pct:.2f}%")
    if equity_change_pct is not None:
        eq_str = f"+{equity_change_pct:.2f}%" if equity_change_pct >= 0 else f"{equity_change_pct:.2f}%"
        lines.append(f"Equity: {eq_str}")

    lines.append("")
    lines.append(f"<i>{BOT_NAME}</i>")

    return send_message("\n".join(lines))


# ============================================
# STATUS ALERTS
# ============================================

def send_bot_started(
    equity: float = None,
    active_positions: int = 0,
    pending_orders: int = 0,
) -> bool:
    """Send notification when bot starts."""
    if not is_enabled():
        return False

    now = datetime.now(TIMEZONE).strftime("%H:%M %d.%m.%Y")

    lines = [
        f"🤖 <b>{BOT_NAME} STARTED</b>",
        "",
        f"Time: {now}",
    ]

    if active_positions > 0:
        lines.append(f"Active Positions: {active_positions}")
    if pending_orders > 0:
        lines.append(f"Pending Orders: {pending_orders}")

    lines.append("")
    lines.append("Ready to trade! 🚀")

    return send_message("\n".join(lines))


def send_bot_stopped(reason: str = "Manual") -> bool:
    """Send notification when bot stops."""
    if not is_enabled():
        return False

    now = datetime.now(TIMEZONE).strftime("%H:%M %d.%m.%Y")

    lines = [
        f"⏹️ <b>{BOT_NAME} STOPPED</b>",
        "",
        f"Time: {now}",
        f"Reason: {reason}",
    ]

    return send_message("\n".join(lines))


def send_error_alert(error: str, context: str = None) -> bool:
    """Send alert for critical errors."""
    if not is_enabled():
        return False

    lines = [
        f"⚠️ <b>ERROR ALERT</b>",
        "",
        f"Error: {error[:200]}",
    ]

    if context:
        lines.append(f"Context: {context}")

    lines.append("")
    lines.append(f"<i>{BOT_NAME}</i>")

    return send_message("\n".join(lines))


# ============================================
# TEST
# ============================================

def send_test() -> bool:
    """Send test message to verify setup."""
    if not is_enabled():
        print("[Telegram] Not configured - set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False

    return send_message(
        f"✅ <b>Test Message</b>\n\n"
        f"Telegram alerts are working!\n\n"
        f"<i>{BOT_NAME}</i>"
    )


if __name__ == '__main__':
    # Test the module
    if send_test():
        print("✅ Telegram test successful!")
    else:
        print("❌ Telegram test failed")
