import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright_bridge import (
    PlaywrightBridge,
    _normalize_url,
    _KEY_MAP,
    is_browser_open,
    navigate,
    click_element,
    type_text,
    press_key,
    scroll,
    read_content,
    close_browser,
)


class TestPlaywrightHelpers(unittest.TestCase):
    def test_normalize_url(self):
        self.assertEqual(_normalize_url("google.com"), "https://google.com")
        self.assertEqual(_normalize_url("http://example.org"), "http://example.org")
        self.assertEqual(_normalize_url("https://formula1.com/en.html"), "https://formula1.com/en.html")
        self.assertEqual(_normalize_url("localhost:8000"), "https://localhost:8000")
        self.assertEqual(_normalize_url("google"), "https://www.google.com")
        self.assertEqual(_normalize_url(""), "")

    def test_key_mapping(self):
        self.assertEqual(_KEY_MAP["return"], "Enter")
        self.assertEqual(_KEY_MAP["enter"], "Enter")
        self.assertEqual(_KEY_MAP["tab"], "Tab")
        self.assertEqual(_KEY_MAP["escape"], "Escape")


class TestPlaywrightIntegration(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        close_browser()

    def test_lifecycle_and_actions(self):
        # Run headless for testing
        bridge = PlaywrightBridge(headless=True)
        self.assertTrue(bridge.start())
        self.assertTrue(bridge.is_active)

        # Navigate to a simple local or data URI HTML page
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Vision Test Page</title></head>
        <body>
            <h1>Vision Agent Playground</h1>
            <button id="btn1">Submit Form</button>
            <input type="text" placeholder="Search query" id="search_input" />
            <a href="#link_target">Documentation</a>
            <div id="output">Initial</div>
            <script>
                document.getElementById('btn1').addEventListener('click', () => {
                    document.getElementById('output').textContent = 'Button Clicked!';
                });
            </script>
        </body>
        </html>
        """
        import base64
        data_url = f"data:text/html;base64,{base64.b64encode(html.encode()).decode()}"

        self.assertTrue(bridge.navigate(data_url))
        self.assertEqual(bridge.current_title, "Vision Test Page")

        # Type into search input
        self.assertTrue(bridge.type_text("Ferrari F1", target="Search query"))

        # Click the button
        self.assertTrue(bridge.click("Submit Form", kind="button"))

        # Verify button click changed output text
        content = bridge.read_content()
        self.assertIn("Button Clicked!", content)
        self.assertIn("Ferrari F1", bridge._page.input_value("#search_input"))

        # Test scroll
        self.assertTrue(bridge.scroll("down", 1))

        # Test key press
        self.assertTrue(bridge.press_key("Enter"))

        bridge.close()
        self.assertFalse(bridge.is_active)


if __name__ == "__main__":
    unittest.main()
