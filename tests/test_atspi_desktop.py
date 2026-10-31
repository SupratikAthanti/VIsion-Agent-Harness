"""
Tests for AT-SPI desktop automation bridge, snapshot/ref system, and routing.

All tests run deterministically without requiring an active X11 desktop or D-Bus session,
using mock accessible trees.
"""

import unittest
from unittest import mock
from typing import List, Optional

import atspi_bridge
from intent_classifier import classify_intent, route_context, is_ref
import agent


class MockAccessibleNode:
    """Mock accessible element tree node for deterministic headless testing."""

    def __init__(
        self,
        name: str,
        role: str,
        bounds=(100, 100, 200, 40),
        children: Optional[List["MockAccessibleNode"]] = None,
        states: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        text: Optional[str] = None,
        value: Optional[str] = None,
        description: str = "",
    ):
        self.name = name
        self.role = role
        self.bounds = bounds
        self.children = children or []
        self.states = states or ["showing", "visible", "enabled"]
        self.actions = actions or []
        self.text = text
        self.value = value
        self.description = description
        self.focused = False

    def get_child_count(self) -> int:
        return len(self.children)

    def get_child_at_index(self, index: int):
        return self.children[index]

    def get_name(self) -> str:
        return self.name

    def get_role_name(self) -> str:
        return self.role

    def get_description(self) -> str:
        return self.description

    def get_bounds(self):
        return self.bounds

    def get_states(self) -> List[str]:
        return self.states

    def get_action_names(self) -> List[str]:
        return self.actions

    def get_text(self, start: int = 0, end: int = -1) -> Optional[str]:
        return self.text

    def get_current_value(self) -> Optional[str]:
        return self.value

    def do_action(self, action_name: str) -> bool:
        return action_name in self.actions

    def set_text_contents(self, new_text: str) -> bool:
        if "editable" in self.states:
            self.text = new_text
            return True
        return False

    def insert_text(self, pos: int, text_to_insert: str) -> bool:
        if "editable" in self.states:
            current = self.text or ""
            self.text = current[:pos] + text_to_insert + current[pos:]
            return True
        return False

    def grab_focus(self) -> bool:
        self.focused = True
        return True


def create_sample_desktop_tree() -> MockAccessibleNode:
    """
    Build a realistic desktop application tree:
    Desktop
      ├── Text Editor (App 0)
      │     └── Main Window
      │           ├── Menu Bar
      │           │     └── File (menu)
      │           │           └── Save (menuitem)
      │           └── Text Area (editable text)
      └── Calculator (App 1)
            └── Calculator Window
                  ├── Display (text / value)
                  └── Button 7 (pushbutton)
    """
    save_item = MockAccessibleNode(
        name="Save",
        role="menuitem",
        bounds=(10, 40, 80, 25),
        actions=["click", "activate"],
    )
    file_menu = MockAccessibleNode(
        name="File",
        role="menu",
        bounds=(10, 10, 50, 25),
        children=[save_item],
        actions=["click", "open"],
    )
    menu_bar = MockAccessibleNode(
        name="Main Menu",
        role="menubar",
        bounds=(0, 0, 800, 30),
        children=[file_menu],
    )
    text_area = MockAccessibleNode(
        name="Document Content",
        role="entry",
        bounds=(10, 50, 780, 500),
        states=["showing", "visible", "enabled", "editable", "focusable"],
        text="Sample file contents",
    )
    text_editor_win = MockAccessibleNode(
        name="Untitled Document - Text Editor",
        role="window",
        bounds=(0, 0, 800, 600),
        children=[menu_bar, text_area],
    )
    text_editor_app = MockAccessibleNode(
        name="Text Editor",
        role="application",
        bounds=(0, 0, 800, 600),
        children=[text_editor_win],
    )

    calc_display = MockAccessibleNode(
        name="Display",
        role="text",
        bounds=(850, 50, 200, 40),
        text="42",
        value="42",
    )
    calc_btn_7 = MockAccessibleNode(
        name="7",
        role="pushbutton",
        bounds=(850, 100, 40, 40),
        actions=["click", "press"],
    )
    calc_win = MockAccessibleNode(
        name="Calculator",
        role="window",
        bounds=(840, 0, 300, 400),
        children=[calc_display, calc_btn_7],
    )
    calc_app = MockAccessibleNode(
        name="Calculator",
        role="application",
        bounds=(840, 0, 300, 400),
        children=[calc_win],
    )

    desktop_root = MockAccessibleNode(
        name="Desktop",
        role="desktop",
        bounds=(0, 0, 1920, 1080),
        children=[text_editor_app, calc_app],
    )
    return desktop_root


class TestAtspiBridge(unittest.TestCase):

    def setUp(self):
        self.mock_desktop = create_sample_desktop_tree()
        atspi_bridge.set_mock_desktop(self.mock_desktop)

    def tearDown(self):
        atspi_bridge.set_mock_desktop(None)

    def test_is_at_spi_available_with_mock(self):
        self.assertTrue(atspi_bridge.is_at_spi_available())

    def test_list_apps(self):
        apps = atspi_bridge.list_apps()
        self.assertEqual(len(apps), 2)
        self.assertEqual(apps[0]["name"], "Text Editor")
        self.assertIn("Untitled Document - Text Editor", apps[0]["windows"])
        self.assertEqual(apps[1]["name"], "Calculator")

    def test_snapshot_app(self):
        snap = atspi_bridge.snapshot("Text Editor", interactive_only=False)
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["app"], "Text Editor")
        self.assertGreater(snap["ref_count"], 0)

        # Check ref IDs formatted @<snapshot_id>:e<N>
        elems = snap["elements"]
        ref_ids = [e["ref_id"] for e in elems]
        self.assertTrue(all(r.startswith(f"@{snap['snapshot_id']}:e") for r in ref_ids))

        # Check element names
        names = [e["name"] for e in elems]
        self.assertIn("Save", names)
        self.assertIn("File", names)
        self.assertIn("Document Content", names)

    def test_resolve_ref(self):
        snap = atspi_bridge.snapshot("Text Editor")
        save_elem = next(e for e in snap["elements"] if e["name"] == "Save")
        ref_id = save_elem["ref_id"]

        resolved = atspi_bridge.resolve_ref(ref_id)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["name"], "Save")
        self.assertEqual(resolved["role"], "menuitem")

    def test_find_element_by_name(self):
        elem = atspi_bridge.find_element_by_name("Save")
        self.assertIsNotNone(elem)
        self.assertEqual(elem["name"], "Save")
        self.assertEqual(elem["role"], "menuitem")

    def test_find_element_by_role(self):
        buttons = atspi_bridge.find_element_by_role("pushbutton")
        self.assertEqual(len(buttons), 1)
        self.assertEqual(buttons[0]["name"], "7")

    def test_click_element_by_ref(self):
        snap = atspi_bridge.snapshot("Text Editor")
        save_elem = next(e for e in snap["elements"] if e["name"] == "Save")
        ref_id = save_elem["ref_id"]

        self.assertTrue(atspi_bridge.click_element(ref_id))

    def test_set_text_contents_editable(self):
        snap = atspi_bridge.snapshot("Text Editor")
        doc_elem = next(e for e in snap["elements"] if e["name"] == "Document Content")
        ref_id = doc_elem["ref_id"]

        ok = atspi_bridge.set_text_contents(ref_id, "New test text contents")
        self.assertTrue(ok)

        # Verify state through read_element
        read_res = atspi_bridge.read_element(ref_id)
        self.assertEqual(read_res["text"], "New test text contents")

    def test_set_text_contents_non_editable_fails(self):
        snap = atspi_bridge.snapshot("Text Editor")
        save_elem = next(e for e in snap["elements"] if e["name"] == "Save")
        ref_id = save_elem["ref_id"]

        ok = atspi_bridge.set_text_contents(ref_id, "Won't work")
        self.assertFalse(ok)

    def test_insert_text(self):
        snap = atspi_bridge.snapshot("Text Editor")
        doc_elem = next(e for e in snap["elements"] if e["name"] == "Document Content")
        ref_id = doc_elem["ref_id"]

        atspi_bridge.set_text_contents(ref_id, "Hello World")
        ok = atspi_bridge.insert_text(ref_id, "Beautiful ", position=6)
        self.assertTrue(ok)

        read_res = atspi_bridge.read_element(ref_id)
        self.assertEqual(read_res["text"], "Hello Beautiful World")

    def test_grab_focus(self):
        snap = atspi_bridge.snapshot("Text Editor")
        doc_elem = next(e for e in snap["elements"] if e["name"] == "Document Content")
        ref_id = doc_elem["ref_id"]

        self.assertTrue(atspi_bridge.grab_focus(ref_id))

    def test_read_element(self):
        calc_elem = atspi_bridge.find_element_by_name("Display", app_name="Calculator")
        self.assertIsNotNone(calc_elem)

        state = atspi_bridge.read_element(calc_elem["ref_id"])
        self.assertEqual(state["name"], "Display")
        self.assertEqual(state["text"], "42")
        self.assertEqual(state["value"], "42")
        self.assertEqual(state["bounds"], (850, 50, 200, 40))


class TestRoutingAndIntent(unittest.TestCase):

    def test_is_ref(self):
        self.assertTrue(is_ref("@s12345678:e1"))
        self.assertTrue(is_ref("@sabcdef01:e42"))
        self.assertTrue(is_ref("@e1"))
        self.assertTrue(is_ref("e1"))
        self.assertFalse(is_ref("Save"))
        self.assertFalse(is_ref("google.com"))

    def test_intent_classification_for_desktop_scopes(self):
        r1 = classify_intent("click File in VS Code")
        self.assertEqual(r1["action"], "CLICK")
        self.assertEqual(r1["target"], "File")
        self.assertEqual(r1["params"].get("app"), "VS Code")

        r2 = classify_intent("type 'hello world' in Document Content in Text Editor")
        self.assertEqual(r2["action"], "TYPE")
        self.assertEqual(r2["target"], "Document Content")
        self.assertEqual(r2["params"].get("app"), "Text Editor")
        self.assertEqual(r2["params"].get("text"), "hello world")

        r3 = classify_intent("list apps")
        self.assertEqual(r3["action"], "LIST_APPS")

        r4 = classify_intent("snapshot Text Editor")
        self.assertEqual(r4["action"], "SNAPSHOT")
        self.assertEqual(r4["params"].get("app"), "Text Editor")

    def test_routing_decisions(self):
        # 1. Explicit browser task
        nav_intent = classify_intent("open google.com")
        self.assertEqual(route_context(nav_intent), "browser")

        search_intent = classify_intent("click Images on the web page")
        self.assertEqual(route_context(search_intent), "browser")

        # 2. Explicit desktop task
        desktop_ref_intent = classify_intent("click @s12345678:e3")
        self.assertEqual(route_context(desktop_ref_intent), "desktop")

        app_scope_intent = classify_intent("click File in VS Code")
        self.assertEqual(route_context(app_scope_intent), "desktop")

        apps_intent = classify_intent("list apps")
        self.assertEqual(route_context(apps_intent), "desktop")

        # 3. Disambiguation with active context
        click_file = classify_intent("click File")
        # Even if browser is open, "File" is desktop OS menu
        self.assertEqual(route_context(click_file, {"browser_open": True}), "desktop")

        # Web-like action when browser is focused
        click_submit = classify_intent("click Submit")
        self.assertEqual(route_context(click_submit, {"browser_open": True}), "browser")
        self.assertEqual(route_context(click_submit, {"focused_app": "Text Editor"}), "desktop")


class TestAgentExecutionIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_desktop = create_sample_desktop_tree()
        atspi_bridge.set_mock_desktop(self.mock_desktop)

    def tearDown(self):
        atspi_bridge.set_mock_desktop(None)

    def test_agent_list_apps_execution(self):
        ok = agent.execute_instruction("list apps", execute=True)
        self.assertTrue(ok)

    def test_agent_snapshot_execution(self):
        ok = agent.execute_instruction("snapshot Text Editor", execute=True)
        self.assertTrue(ok)

    def test_agent_click_in_app_execution(self):
        ok = agent.execute_instruction("click Save in Text Editor", execute=True)
        self.assertTrue(ok)

    def test_agent_type_in_app_execution(self):
        ok = agent.execute_instruction("type 'Testing AT-SPI' in Document Content in Text Editor", execute=True)
        self.assertTrue(ok)

        # Check that text was written to the mock element
        elem = atspi_bridge.find_element_by_name("Document Content")
        self.assertEqual(elem["text"], "Testing AT-SPI")

    def test_agent_read_desktop_execution(self):
        ok = agent.execute_instruction("read Display in Calculator", execute=True)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
