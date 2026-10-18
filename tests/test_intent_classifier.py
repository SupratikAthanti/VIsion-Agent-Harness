import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intent_classifier import classify_intent  # noqa: E402


class SpecExamples(unittest.TestCase):
    def test_click_with_kind_word_stripped(self):
        r = classify_intent("Click the gemini button")
        self.assertEqual(r["action"], "CLICK")
        self.assertEqual(r["target"], "gemini")
        self.assertEqual(r["params"]["kind"], "button")

    def test_type_into_field(self):
        r = classify_intent("Type hello in the search box")
        self.assertEqual((r["action"], r["target"], r["params"]["text"]), ("TYPE", "search box", "hello"))

    def test_navigate_url(self):
        r = classify_intent("Go to youtube.com")
        self.assertEqual((r["action"], r["params"]["url"]), ("NAVIGATE", "youtube.com"))

    def test_scroll_direction(self):
        r = classify_intent("Scroll down")
        self.assertEqual((r["action"], r["params"]["direction"]), ("SCROLL", "down"))

    def test_find_menu_is_click(self):
        r = classify_intent("Find the settings menu")
        self.assertEqual((r["action"], r["target"]), ("CLICK", "settings menu"))


class Variations(unittest.TestCase):
    def test_press_and_tap_verbs(self):
        self.assertEqual(classify_intent("press submit")["target"], "submit")
        self.assertEqual(classify_intent("tap search")["target"], "search")

    def test_quoted_text(self):
        r = classify_intent('type "hi there" into the box')
        self.assertEqual((r["params"]["text"], r["target"]), ("hi there", "box"))

    def test_type_without_target(self):
        r = classify_intent("type hello")
        self.assertEqual((r["action"], r["target"], r["params"]["text"]), ("TYPE", None, "hello"))

    def test_press_enter_is_key(self):
        r = classify_intent("press enter")
        self.assertEqual((r["action"], r["params"]["key"]), ("TYPE", "Return"))

    def test_wait_units(self):
        self.assertEqual(classify_intent("wait 2 seconds")["params"]["seconds"], 2.0)
        self.assertAlmostEqual(classify_intent("wait 500 ms")["params"]["seconds"], 0.5)
        self.assertEqual(classify_intent("wait")["params"]["seconds"], 1.0)

    def test_scroll_up_with_amount(self):
        r = classify_intent("scroll up 3")
        self.assertEqual((r["params"]["direction"], r["params"]["amount"]), ("up", 3))

    def test_drag(self):
        r = classify_intent("drag the file to the trash")
        self.assertEqual((r["action"], r["target"], r["params"]["destination"]), ("DRAG", "file", "trash"))

    def test_where_is_is_locate_only(self):
        r = classify_intent("Where is the Ask Gemini button?")
        self.assertEqual(r["action"], "CLICK")
        self.assertTrue(r["params"]["locate_only"])
        self.assertEqual(r["target"], "Ask Gemini")

    def test_open_name_is_click(self):
        r = classify_intent("open settings")
        self.assertEqual((r["action"], r["target"]), ("CLICK", "settings"))

    def test_bare_element_name_is_click(self):
        self.assertEqual(classify_intent("the gemini button")["action"], "CLICK")


class Multistep(unittest.TestCase):
    def test_then_split(self):
        r = classify_intent("click login then type bob")
        self.assertEqual(r["action"], "MULTISTEP")
        self.assertEqual([s["action"] for s in r["params"]["steps"]], ["CLICK", "TYPE"])

    def test_and_then_split(self):
        r = classify_intent("type hello and then press enter")
        self.assertEqual([s["action"] for s in r["params"]["steps"]], ["TYPE", "TYPE"])

    def test_and_without_verbs_is_not_split(self):
        r = classify_intent("type salt and pepper in the box")
        self.assertEqual(r["action"], "TYPE")
        self.assertEqual(r["params"]["text"], "salt and pepper")


class Ambiguity(unittest.TestCase):
    def test_click_it_needs_clarification(self):
        r = classify_intent("Click it")
        self.assertEqual(r["action"], "CLICK")
        self.assertIsNone(r["target"])
        self.assertIsNotNone(r["needs_clarification"])

    def test_bare_click_needs_clarification(self):
        self.assertIsNotNone(classify_intent("click")["needs_clarification"])

    def test_navigate_without_url_asks_for_one(self):
        r = classify_intent("navigate to youtube")
        self.assertIsNone(r["params"]["url"])
        self.assertIn("full address", r["needs_clarification"])

    def test_gibberish_is_unknown(self):
        r = classify_intent("blorp the flarn")
        self.assertEqual(r["action"], "UNKNOWN")
        self.assertIsNotNone(r["needs_clarification"])

    def test_empty(self):
        self.assertEqual(classify_intent("   ")["action"], "UNKNOWN")


class Performance(unittest.TestCase):
    def test_under_50ms(self):
        queries = ["Click the gemini button", "Type hello in the search box", "Go to youtube.com",
                   "click login then type bob", "scroll down", "where is the Ask Gemini button?"]
        worst = 0.0
        for q in queries:
            t0 = time.perf_counter()
            classify_intent(q)
            worst = max(worst, time.perf_counter() - t0)
        self.assertLess(worst, 0.05)


if __name__ == "__main__":
    unittest.main()
