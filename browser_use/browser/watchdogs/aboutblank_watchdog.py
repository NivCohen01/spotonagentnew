"""About:blank watchdog for managing about:blank tabs with a neutral loading overlay."""

from typing import TYPE_CHECKING, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.target import TargetID
from pydantic import PrivateAttr

from browser_use.browser.events import (
    AboutBlankDVDScreensaverShownEvent,
    BrowserStopEvent,
    BrowserStoppedEvent,
    CloseTabEvent,
    NavigateToUrlEvent,
    TabClosedEvent,
    TabCreatedEvent,
)
from browser_use.browser.watchdog_base import BaseWatchdog

if TYPE_CHECKING:
    pass


class AboutBlankWatchdog(BaseWatchdog):
    """Ensures there's always exactly one about:blank tab with a loading overlay."""

    # Event contracts
    LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [
        BrowserStopEvent,
        BrowserStoppedEvent,
        TabCreatedEvent,
        TabClosedEvent,
    ]
    EMITS: ClassVar[list[type[BaseEvent]]] = [
        NavigateToUrlEvent,
        CloseTabEvent,
        AboutBlankDVDScreensaverShownEvent,
    ]

    _stopping: bool = PrivateAttr(default=False)

    async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
        """Handle browser stop request - stop creating new tabs."""
        # logger.info('[AboutBlankWatchdog] Browser stop requested, stopping tab creation')
        self._stopping = True

    async def on_BrowserStoppedEvent(self, event: BrowserStoppedEvent) -> None:
        """Handle browser stopped event."""
        # logger.info('[AboutBlankWatchdog] Browser stopped')
        self._stopping = True

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Check tabs when a new tab is created."""
        # logger.debug(f'[AboutBlankWatchdog] New tab created: {event.url}')

        # If an about:blank tab was created, show loading overlay on all about:blank tabs
        if event.url == "about:blank":
            await self._show_dvd_screensaver_on_about_blank_tabs()

    async def on_TabClosedEvent(self, event: TabClosedEvent) -> None:
        """Check tabs when a tab is closed and proactively create about:blank if needed."""
        # logger.debug('[AboutBlankWatchdog] Tab closing, checking if we need to create about:blank tab')

        # Don't create new tabs if browser is shutting down
        if self._stopping:
            # logger.debug('[AboutBlankWatchdog] Browser is stopping, not creating new tabs')
            return

        # Check if we're about to close the last tab (event happens BEFORE tab closes)
        # Use _cdp_get_all_pages for quick check without fetching titles
        page_targets = await self.browser_session._cdp_get_all_pages()
        if len(page_targets) <= 1:
            self.logger.debug(
                "[AboutBlankWatchdog] Last tab closing, creating new about:blank tab to avoid closing entire browser"
            )
            # Create the animation tab since no tabs should remain
            navigate_event = self.event_bus.dispatch(NavigateToUrlEvent(url="about:blank", new_tab=True))
            await navigate_event
            # Show loading overlay on the new tab
            await self._show_dvd_screensaver_on_about_blank_tabs()
        else:
            # Multiple tabs exist, check after close
            await self._check_and_ensure_about_blank_tab()

    async def attach_to_target(self, target_id: TargetID) -> None:
        """AboutBlankWatchdog doesn't monitor individual targets."""
        pass

    async def _check_and_ensure_about_blank_tab(self) -> None:
        """Check current tabs and ensure exactly one about:blank tab with animation exists."""
        try:
            # For quick checks, just get page targets without titles to reduce noise
            page_targets = await self.browser_session._cdp_get_all_pages()

            # If no tabs exist at all, create one to keep browser alive
            if len(page_targets) == 0:
                # Only create a new tab if there are no tabs at all
                self.logger.debug("[AboutBlankWatchdog] No tabs exist, creating new about:blank loading overlay tab")
                navigate_event = self.event_bus.dispatch(NavigateToUrlEvent(url="about:blank", new_tab=True))
                await navigate_event
                # Show loading overlay on the new tab
                await self._show_dvd_screensaver_on_about_blank_tabs()
            # Otherwise there are tabs, don't create new ones to avoid interfering

        except Exception as e:
            self.logger.error(f"[AboutBlankWatchdog] Error ensuring about:blank tab: {e}")

    async def _show_dvd_screensaver_on_about_blank_tabs(self) -> None:
        """Show loading overlay on all about:blank pages only."""
        try:
            # Get just the page targets without expensive title fetching
            page_targets = await self.browser_session._cdp_get_all_pages()
            browser_session_label = str(self.browser_session.id)[-4:]

            for page_target in page_targets:
                target_id = page_target["targetId"]
                url = page_target["url"]

                # Only target about:blank pages specifically
                if url == "about:blank":
                    await self._show_dvd_screensaver_loading_animation_cdp(target_id, browser_session_label)

        except Exception as e:
            self.logger.error(f"[AboutBlankWatchdog] Error showing loading overlay: {e}")

    async def _show_dvd_screensaver_loading_animation_cdp(
        self, target_id: TargetID, browser_session_label: str
    ) -> None:
        """
        Injects a neutral, professional loading overlay into the target using CDP.
        This is used to visually indicate that the browser is setting up or waiting without branding.
        """
        try:
            # Create temporary session for this target without switching focus
            temp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=False)

            # Inject the loading overlay script (idempotent)
            script = f"""
                (function(browser_session_label) {{
                    const boot = () => {{
                        if (window.__neutralLoadingOverlayActive) {{
                            return;
                        }}

                        if (document.getElementById('neutral-loading-overlay')) {{
                            window.__neutralLoadingOverlayActive = true;
                            return;
                        }}

                        if (!document.body) {{
                            if (document.readyState === 'loading') {{
                                document.addEventListener('DOMContentLoaded', boot, {{ once: true }});
                            }}
                            return;
                        }}

                        window.__neutralLoadingOverlayActive = true;

                        const animated_title = `Preparing session ${{browser_session_label}}...`;
                        document.title = animated_title;

                        const overlay = document.createElement('div');
                        overlay.id = 'neutral-loading-overlay';
                        overlay.setAttribute('role', 'presentation');
                        overlay.style.position = 'fixed';
                        overlay.style.inset = '0';
                        overlay.style.display = 'flex';
                        overlay.style.alignItems = 'center';
                        overlay.style.justifyContent = 'center';
                        overlay.style.background =
                            'radial-gradient(circle at 20% 20%, rgba(255,255,255,0.04), transparent 35%), ' +
                            'radial-gradient(circle at 80% 30%, rgba(255,255,255,0.04), transparent 30%), ' +
                            'linear-gradient(135deg, #0f172a 0%, #111827 45%, #0b1224 100%)';
                        overlay.style.zIndex = '99999';
                        overlay.style.color = '#e5e7eb';
                        overlay.style.fontFamily = 'Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif';

                        const panel = document.createElement('div');
                        panel.style.padding = '32px 36px';
                        panel.style.background = 'rgba(17, 24, 39, 0.72)';
                        panel.style.border = '1px solid rgba(255, 255, 255, 0.08)';
                        panel.style.backdropFilter = 'blur(4px)';
                        panel.style.borderRadius = '16px';
                        panel.style.boxShadow = '0 20px 45px rgba(0, 0, 0, 0.35)';
                        panel.style.display = 'grid';
                        panel.style.gap = '16px';
                        panel.style.minWidth = '320px';

                        const spinner = document.createElement('div');
                        spinner.className = 'neutral-spinner';

                        const dot = document.createElement('div');
                        dot.className = 'neutral-dot';
                        spinner.appendChild(dot);

                        const title = document.createElement('div');
                        title.textContent = 'Preparing your workspace';
                        title.style.fontSize = '18px';
                        title.style.fontWeight = '600';
                        title.style.letterSpacing = '0.2px';

                        const subtitle = document.createElement('div');
                        subtitle.textContent = 'Launching a clean tab and getting things ready.';
                        subtitle.style.fontSize = '14px';
                        subtitle.style.color = '#9ca3af';

                        const sessionLabel = document.createElement('div');
                        sessionLabel.textContent = `Session ${{browser_session_label}}`;
                        sessionLabel.style.fontSize = '12px';
                        sessionLabel.style.textTransform = 'uppercase';
                        sessionLabel.style.letterSpacing = '1.2px';
                        sessionLabel.style.color = '#6b7280';

                        panel.appendChild(spinner);
                        panel.appendChild(title);
                        panel.appendChild(subtitle);
                        panel.appendChild(sessionLabel);
                        overlay.appendChild(panel);
                        document.body.appendChild(overlay);

                        const style = document.createElement('style');
                        style.textContent = `
                            #neutral-loading-overlay .neutral-spinner {{
                                position: relative;
                                width: 64px;
                                height: 64px;
                                margin: 0 auto;
                            }}
                            #neutral-loading-overlay .neutral-spinner::before,
                            #neutral-loading-overlay .neutral-spinner::after {{
                                content: '';
                                position: absolute;
                                inset: 0;
                                border-radius: 50%;
                                border: 3px solid transparent;
                                border-top-color: #60a5fa;
                                border-right-color: #60a5fa;
                                animation: spinPulse 1.4s linear infinite;
                                opacity: 0.9;
                            }}
                            #neutral-loading-overlay .neutral-spinner::after {{
                                inset: 8px;
                                border-color: transparent;
                                border-bottom-color: #c084fc;
                                border-left-color: #c084fc;
                                animation-direction: reverse;
                                animation-duration: 1.1s;
                                opacity: 0.8;
                            }}
                            #neutral-loading-overlay .neutral-spinner .neutral-dot {{
                                position: absolute;
                                inset: 22px;
                                border-radius: 50%;
                                background: linear-gradient(135deg, #60a5fa, #a855f7);
                                box-shadow: 0 8px 30px rgba(96, 165, 250, 0.35);
                                animation: pulse 1.6s ease-in-out infinite;
                            }}
                            @keyframes spinPulse {{
                                from {{ transform: rotate(0deg); }}
                                to {{ transform: rotate(360deg); }}
                            }}
                            @keyframes pulse {{
                                0% {{ transform: scale(0.95); opacity: 0.85; }}
                                50% {{ transform: scale(1.05); opacity: 1; }}
                                100% {{ transform: scale(0.95); opacity: 0.85; }}
                            }}
                        `;
                        document.head.appendChild(style);
                    }};

                    boot();
                }})('{browser_session_label}');
            """

            await temp_session.cdp_client.send.Runtime.evaluate(params={"expression": script}, session_id=temp_session.session_id)

            # No need to detach - session is cached

            # Dispatch event
            self.event_bus.dispatch(AboutBlankDVDScreensaverShownEvent(target_id=target_id))

        except Exception as e:
            self.logger.error(f"[AboutBlankWatchdog] Error injecting loading overlay: {e}")
