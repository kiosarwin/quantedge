"""
Ninja Trader Watchdog — Maintenance Engine

Monitors the bot process and auto-restarts it when:
  - Process dies unexpectedly
  - Bot hangs (no log activity for > HANG_TIMEOUT seconds)
  - Memory usage exceeds threshold

Sends Telegram alerts with diagnosis on every intervention.
Runs as a separate process alongside the bot.
"""
import os
import time
import signal
import subprocess
import logging
import httpx
from pathlib import Path
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
BOT_DIR       = Path("/home/kiosarwin/ninja_trader")
BOT_CMD       = [str(BOT_DIR / "venv/bin/python"), "-m", "src"]
LOG_FILE      = BOT_DIR / "logs/futures_trader.log"
WATCHDOG_LOG  = BOT_DIR / "logs/watchdog.log"

load_dotenv(BOT_DIR / ".env")

CHECK_INTERVAL   = 30
HANG_TIMEOUT     = 300
RESTART_COOLDOWN = 15
MAX_RESTARTS     = 10
RESTART_WINDOW   = 3600
MAX_MEMORY_MB    = float(os.getenv("WATCHDOG_MAX_MEMORY_MB", "1400"))

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_URL     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(WATCHDOG_LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("watchdog")


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(TELEGRAM_URL, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            })
            if resp.status_code != 200:
                log.warning("Watchdog Telegram send failed with Markdown: %s %s", resp.status_code, resp.text[:200])
                fallback = client.post(TELEGRAM_URL, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": msg,
                    "disable_web_page_preview": True,
                })
                if fallback.status_code != 200:
                    log.warning(
                        "Watchdog Telegram plain-text fallback failed: %s %s",
                        fallback.status_code,
                        fallback.text[:200],
                    )
    except Exception as e:
        log.warning("Telegram alert failed: %s", e)


# ── Log diagnosis ─────────────────────────────────────────────────────────────
_EXTERNAL_KEYWORDS = (
    "ExchangeNotAvailable", "NetworkError", "RequestTimeout",
    "SSLCertVerificationError", "ClientConnectorCertificateError",
    "Hostname mismatch", "ssl:", "SSL:", "ConnectionRefusedError",
    "aiohttp.client_exceptions",
)

def extract_last_errors(n_tail: int = 80) -> tuple[str, str]:
    """
    Reads last n_tail lines of the bot log.
    Returns (diagnosis_text, category) where category is 'external' or 'bot_bug'.
    """
    try:
        lines = LOG_FILE.read_text(errors="replace").splitlines()[-n_tail:]
    except Exception:
        return "_(could not read log file)_", "unknown"

    # Collect meaningful lines: errors, exceptions, tracebacks
    hits = []
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped for kw in (
            "ERROR", "WARNING", "Traceback", "Exception", "Error:",
            "CRITICAL", "raise ", "assert ",
        )):
            # Strip rich markup and timestamps for readability
            clean = stripped
            for prefix in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
                if prefix in clean:
                    clean = clean[clean.index(prefix):]
                    break
            hits.append(clean[:120])

    if not hits:
        # No errors — just show the last 5 lines
        hits = [l.strip()[:120] for l in lines[-5:] if l.strip()]

    diagnosis = "\n".join(hits[-8:])  # last 8 relevant lines

    # Classify: external or bot bug
    full_tail = "\n".join(lines)
    is_external = any(kw in full_tail for kw in _EXTERNAL_KEYWORDS)
    category = "external" if is_external else "bot_bug"

    return diagnosis, category


def format_alert(title: str, restart_num: int, extra: str, diagnosis: str, category: str) -> str:
    if category == "external":
        icon = "⚡"
        verdict = "_Looks like an external issue (Binance/network). Bot will self-recover._"
    else:
        icon = "🚨"
        verdict = "_Looks like a bot error. May need manual inspection._"

    msg = (
        f"{icon} *Watchdog: {title}* (#{restart_num})\n"
        f"{extra}"
        f"──────────────────────\n"
        f"*Last log entries:*\n"
        f"```\n{diagnosis[:600]}\n```\n"
        f"──────────────────────\n"
        f"{verdict}"
    )
    return msg


# ── Process helpers ───────────────────────────────────────────────────────────
def start_bot() -> subprocess.Popen:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_FILE, "a")
    proc = subprocess.Popen(
        BOT_CMD,
        cwd=BOT_DIR,
        stdout=log_fh,
        stderr=log_fh,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    log.info("Bot started — PID %d", proc.pid)
    return proc


def kill_bot(proc: subprocess.Popen) -> None:
    try:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def log_last_modified_ago() -> float:
    try:
        return time.time() - LOG_FILE.stat().st_mtime
    except FileNotFoundError:
        return float("inf")


def memory_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


# ── Watchdog loop ─────────────────────────────────────────────────────────────
def run() -> None:
    log.info("Watchdog started — monitoring Ninja Trader bot")
    send_telegram(
        "🛡 *Watchdog online — Maintenance Engine active*\n"
        "I monitor the bot 24/7. If it crashes or hangs, I restart it and tell you exactly why."
    )

    proc: subprocess.Popen | None = None
    restart_times: list[float] = []
    restart_count = 0
    proc_start_time = time.time()

    while True:
        try:
            now = time.time()

            # ── Start bot if not running ──────────────────────────────
            if proc is None or proc.poll() is not None:
                exit_code = proc.poll() if proc else None
                if proc is not None:
                    log.warning("Bot exited with code %s — restarting", exit_code)

                # Enforce max restart rate
                restart_times = [t for t in restart_times if now - t < RESTART_WINDOW]
                if len(restart_times) >= MAX_RESTARTS:
                    diagnosis, category = extract_last_errors()
                    msg = (
                        f"🚨 *Watchdog: Too many restarts* ({MAX_RESTARTS}x in 1 hour)\n"
                        f"*Pausing — manual intervention needed.*\n"
                        f"──────────────────────\n"
                        f"*What I last saw:*\n"
                        f"```\n{diagnosis[:600]}\n```"
                    )
                    log.error("Too many restarts — pausing")
                    send_telegram(msg)
                    time.sleep(RESTART_COOLDOWN * 10)
                    restart_times.clear()
                    continue

                # Grab diagnosis BEFORE restarting (log still has the crash reason)
                if exit_code is not None:
                    diagnosis, category = extract_last_errors()

                time.sleep(RESTART_COOLDOWN)
                proc = start_bot()
                restart_count += 1
                restart_times.append(now)
                proc_start_time = now
                LOG_FILE.touch()

                if exit_code is not None:
                    alert = format_alert(
                        title="Bot crashed & restarted",
                        restart_num=restart_count,
                        extra=f"Exit code: `{exit_code}`  |  New PID: `{proc.pid}`\n",
                        diagnosis=diagnosis,
                        category=category,
                    )
                    send_telegram(alert)
                continue

            # ── Grace period after fresh start ────────────────────────
            if now - proc_start_time < 60:
                time.sleep(CHECK_INTERVAL)
                continue

            # ── Check for hung process ────────────────────────────────
            silence = log_last_modified_ago()
            if silence > HANG_TIMEOUT:
                log.warning("Bot hung — no log activity for %.0fs — killing", silence)
                diagnosis, category = extract_last_errors()
                alert = format_alert(
                    title="Bot hung & restarted",
                    restart_num=restart_count + 1,
                    extra=f"No log activity for `{silence:.0f}s`.\n",
                    diagnosis=diagnosis,
                    category=category,
                )
                send_telegram(alert)
                kill_bot(proc)
                proc = None
                continue

            # ── Memory check ──────────────────────────────────────────
            mem = memory_mb(proc.pid)
            if mem > MAX_MEMORY_MB:
                log.warning("Bot using %.0f MB RAM — restarting", mem)
                send_telegram(
                    f"⚠️ *Watchdog: High memory — restarting* (#{restart_count + 1})\n"
                    f"Bot using `{mem:.0f} MB` RAM (limit `{MAX_MEMORY_MB:.0f} MB`). Killing to free memory.\n"
                    f"_This is usually a memory leak after many hours of running._"
                )
                kill_bot(proc)
                proc = None
                continue

            log.info(
                "Bot healthy — PID %d | uptime %.0fs | log silence %.0fs | mem %.0fMB",
                proc.pid, now - (restart_times[-1] if restart_times else now),
                silence, mem,
            )

        except Exception as e:
            log.error("Watchdog error: %s", e)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    env_file = BOT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_URL     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    run()
