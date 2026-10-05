"""The content checks of the skill tools (``mirix/services/skill_text_checks.py``).

Pure CPU, offline. The tools' use of them is tested against a database in
``test_skill_curator_tools.py`` (TestContentChecks, TestCitationsNeedAddedText,
TestEditFeedback).
"""

import time

from mirix.services.skill_text_checks import (
    ECHO_MAX_WORDS,
    LINE_DIFF_MAX_CELLS,
    SKILL_CITATION_MIN_ADDED_WORDS,
    added_words,
    content_words,
    dropped_sentences,
    numbering_problems,
    placeholder_problem,
)

NAME_DOC = "Lowercase words joined by hyphens that name the task and its approach."
INSTRUCTIONS_DOC = "Detailed instructions as plain text: the steps and caveats."


class TestContentWords:
    def test_words_need_a_letter_and_cjk_characters_count_one_each(self):
        assert content_words("Open it. 1. 2) --- 42") == ["Open", "it."]
        assert content_words("  ") == [] and content_words(None) == []
        assert content_words("部署到生产") == ["部", "署", "到", "生", "产"]
        assert content_words("run部署") == ["部", "署", "run"]


class TestLegitimateContentPasses:
    """The checks apply to every caller of the skill tools,
    so real content must pass; each value below was rejected."""

    def test_a_script_without_spaces_has_words(self):
        thai = "ขั้นตอนการปรับใช้ระบบไปยังเซิร์ฟเวอร์จริงอย่างปลอดภัย"
        assert len(content_words(thai)) >= 3
        assert placeholder_problem("description", thai, min_words=3) is None
        khmer = "ការដាក់ឱ្យប្រើប្រាស់ប្រព័ន្ធដោយសុវត្ថិភាព"
        assert placeholder_problem("description", khmer, min_words=3) is None
        # Latin and CJK are counted as before.
        assert content_words("ok ไทย 部署")[0] == "ok"
        assert content_words("ok ไทย 部署")[-2:] == ["部", "署"]

    def test_a_topic_that_merely_mentions_a_marker_is_content(self):
        for field, value in (
            ("description", "TODO list triage: sort open items by due date and owner."),
            (
                "description",
                "Save a draft email and complete it later from the outbox.",
            ),
            (
                "description",
                "Fill a placeholder description field in the CMS before publishing.",
            ),
            (
                "description",
                "Dates to be decided later are marked with a question mark.",
            ),
        ):
            assert placeholder_problem(field, value, min_words=3) is None, value

    def test_domain_words_are_real_names_and_triggers(self):
        for field, value in (
            ("triggers[0]", "test"),
            ("triggers[0]", "json"),
            ("name", "test"),
            ("name", "json"),
            ("triggers[0]", "list"),
            ("triggers[0]", "number"),
        ):
            assert placeholder_problem(field, value) is None, (field, value)
        # True fillers still are placeholders.
        for value in ("string", "none", "n/a", "todo", "object"):
            assert "is a placeholder" in placeholder_problem("triggers[0]", value), (
                value
            )

    def test_one_word_of_the_parameter_description_is_no_echo(self):
        doc = "what was done or answered"
        assert placeholder_problem("examples[0].output", "done", doc) is None
        assert "repeats the words" in placeholder_problem(
            "examples[0].output", "what was done", doc
        )

    def test_self_declared_placeholders_are_still_rejected(self):
        for value in (
            "TODO: write the steps",
            "TBD",
            "FIXME - steps come later",
            "To be filled in later.",
            "Details to be added later.",
            "Fill it in later, once the tool is known.",
            "A placeholder entry for the deploy steps",
            "This is a placeholder description of the flow.",
        ):
            assert placeholder_problem("instructions", value), value


class TestPlaceholderProblem:
    def test_too_few_words(self):
        assert "has 1 word(s)" in placeholder_problem(
            "description", "deploy", min_words=3
        )
        assert "has 0 word(s)" in placeholder_problem("description", "...", min_words=3)
        assert (
            placeholder_problem("description", "Retry failed uploads", min_words=3)
            is None
        )

    def test_generic_placeholders_and_the_field_name(self):
        for value in ("string", "TEXT", "n/a", "...", "'None'", "lorem ipsum", "name"):
            assert "is a placeholder" in placeholder_problem("name", value), value
        assert "is a placeholder" in placeholder_problem("examples[0].input", "input")

    def test_a_schema_echo(self):
        assert "repeats the words" in placeholder_problem(
            "name", "kebab-case name", NAME_DOC
        )
        assert "repeats the words" in placeholder_problem(
            "instructions", "detailed plain text", INSTRUCTIONS_DOC
        )
        assert "repeats the words" in placeholder_problem(
            "name", "task-approach", NAME_DOC
        )
        # A word the schema does not use makes it content.
        assert placeholder_problem("name", "verify-against-source", NAME_DOC) is None
        assert placeholder_problem("name", "json-output", NAME_DOC) is None
        # Only short values can be echoes.
        long_echo = " ".join(["detailed plain text"] * ECHO_MAX_WORDS)
        assert placeholder_problem("instructions", long_echo, INSTRUCTIONS_DOC) is None
        # An index in the field's label is no vocabulary.
        assert placeholder_problem("examples[0].output", "0") is None

    def test_a_value_that_says_it_is_a_placeholder(self):
        for value in (
            "This placeholder skill was created in error. Please delete it.",
            "Placeholder for the steps.",
            "To be filled in later.",
            "TODO: write the steps",
            "We will expand this later",
        ):
            assert "says it is a placeholder" in (
                placeholder_problem("instructions", value) or ""
            ), value
        for value in (
            "Replace the placeholder text with the customer name.",
            "Placeholder values in a template must be replaced before sending.",
            "Scan the files for TODO comments and list them.",
        ):
            assert placeholder_problem("instructions", value) is None, value


class TestAddedWords:
    def test_what_an_edit_adds(self):
        assert added_words(
            "1. Open it.\n5. Check it.", "1. Open it.\n6. Check it."
        ) == (0, 0)
        assert added_words("Open it. Check it.", "Open   it.\n\nCheck it.") == (0, 0)
        assert added_words("Open it. Check it.", "Open it.") == (0, 2)
        assert added_words("Trust the message.", "Never trust the message.") == (1, 0)
        assert added_words("Check the cup.", "Check the mug.") == (1, 1)
        assert SKILL_CITATION_MIN_ADDED_WORDS == 1

    def test_numbers_are_words_but_list_markers_are_not(self):
        """A changed number changes what a skill says (a retry count, a port,
        a year, a sign); only a list marker is a numbering."""
        assert added_words(
            "Retry a failed upload 3 times, waiting 30 seconds.",
            "Retry a failed upload 5 times, waiting 120 seconds.",
        ) == (2, 2)
        for old, new in (
            ("Use port 8080.", "Use port 8531."),
            ("Filter to 2019.", "Filter to 2020."),
            ("Multiply by 1000.", "Multiply by 1000000."),
            ("Shift the clock +8 hours.", "Shift the clock -8 hours."),
            ("Retry on error 404.", "Retry on error 500."),
        ):
            assert added_words(old, new) == (1, 1), (old, new)
        # Renumbered list items (any marker style) and a decimal kept as is.
        assert added_words(
            "1. Weigh 3.5 kg.\n2) Stir.\n- Serve.", "4. Weigh 3.5 kg.\n9) Stir.\n* Serve."
        ) == (0, 0)

    def test_a_moved_line_adds_its_words_and_a_reflow_adds_none(self):
        old = "1. Open the file.\n2. Check the unit.\n3. Convert the value."
        moved = "1. Check the unit.\n2. Open the file.\n3. Convert the value."
        added, removed = added_words(old, moved)
        assert added >= 3 and removed >= 3
        assert added_words(
            "Open the file and check the unit before converting.",
            "Open the file\nand check the unit\nbefore converting.",
        ) == (0, 0)
        # Only the changed lines count, wherever they are.
        body = "\n".join(f"Step about topic {i}." for i in range(50))
        assert added_words(body, body.replace("topic 7.", "topic 7 twice.")) == (1, 0)

    def test_a_phrase_loop_is_diffed_in_linear_time(self):
        """The skill tools diff inline on the event loop. A word-level
        difflib diff of a 12k-character phrase loop took 20 s (and 185 s for
        "the " x 3000); every case here must take milliseconds."""
        cases = [
            ("check the value. " * 700, "check the value and the unit. " * 400),
            ("the " * 3000, "the a " * 3000),
            ("check the value.\n" * 700, "check the value and the unit.\n" * 400),
            ("the\n" * 3000, "the\na\n" * 3000),
            ("x\ny\n" * 1500, "x\nz\n" * 1500),
        ]
        for old, new in cases:
            start = time.perf_counter()
            added, removed = added_words(old, new)
            elapsed = time.perf_counter() - start
            assert elapsed < 1.0, (old[:20], new[:20], elapsed)
            assert added > 0
        assert added_words("the " * 3000, "the a " * 3000) == (3000, 0)
        # The line alignment is capped: a region over the cap is one block.
        assert LINE_DIFF_MAX_CELLS == 40_000
        assert added_words("x\ny\n" * 1500, "x\nz\n" * 1500) == (1500, 1500)

    def test_a_reordering_adds_the_words_it_moves(self):
        """A changed block was compared as a bag of words, so an
        edit that swaps words within a line or across the lines of the block
        counted 0 added words and its citations were dropped. Each word whose
        position changed now counts (a reflow still counts none)."""
        swapped = added_words("Run lint before tests.", "Run tests before lint.")
        assert swapped == (2, 2)
        old = "1. Use tool alpha for tables\n2. Use tool beta for text"
        for new in (
            "1. Use tool beta for tables\n2. Use tool alpha for text",
            "1. Use tool alpha for text\n2. Use tool beta for tables",
        ):
            assert added_words(old, new) == (2, 2), new
        # A swap inside a region of unchanged lines, and one next to a reflow.
        body = "\n".join(f"Step about topic {i}." for i in range(20))
        swapped = added_words(body, body.replace("about topic 7", "topic about 7"))
        assert swapped == (2, 2)
        assert added_words(
            "Open the file and check the unit before converting.",
            "Open the unit\nand check the file before converting.",
        ) == (2, 2)
        # The same words in the same order, only on other lines: a reflow.
        assert added_words(
            "Open the file and check\nthe unit before converting.",
            "Open the file\nand check the unit\nbefore converting.",
        ) == (0, 0)
        # A reordered block that also adds words counts the added ones.
        grown = added_words("Run lint before tests.", "Run tests before lint twice.")
        assert grown == (1, 0)

    def test_a_step_prefix_is_a_numbering(self):
        """A rename from "Step 1:" to "Step 2:" counted 1 added word and earned
        citation credit; a "Step N" prefix is numbered like a list marker."""
        old = "Step 1: Open the file.\nStep 2: Check the unit."
        new = "Step 2: Open the file.\nStep 3: Check the unit."
        assert added_words(old, new) == (0, 0)
        for before, after in (
            ("Step 1. Open it.", "Step 4. Open it."),
            ("step 1 - Open it.", "Step 12 — Open it."),
            ("**Step 1:** Open it.", "**Step 2:** Open it."),
            ("- Step 1: Open it.", "3. Step 2: Open it."),
            ("### Step 1\nOpen it.", "### Step 2\nOpen it."),
            ("STEP 3) Open it.", "STEP 5) Open it."),
            ("Step 1: Open it.", "Open it."),
        ):
            assert added_words(before, after) == (0, 0), (before, after)
        # A step named inside a sentence, or a "Step N" with no separator, is
        # content: it points at another step.
        for before, after in (
            ("Retry step 3 twice.", "Retry step 4 twice."),
            ("Step 3 of the build fails.", "Step 4 of the build fails."),
            ("Step 1.5 kg of flour.", "Step 2.5 kg of flour."),
        ):
            assert added_words(before, after) == (1, 1), (before, after)
        # The renumbered steps still get the rewrite report's sentence match.
        assert dropped_sentences(
            "Step 1: Open the config file first.", "Step 2: Open the config file first."
        ) == ([], 1)

    def test_every_common_numbering_style_is_a_numbering(self):
        """Only "N.", "N)", bullets, headings and "Step N" were
        numbering, so renumbering "2.1." -> "3.1.", "(1)" -> "(2)", "a." ->
        "b.", "i." -> "ii.", "Phase 1:" -> "Phase 2:" or "第1步：" ->
        "第2步：" counted 1 added word and kept its citation."""
        for before, after in (
            ("2.1. Install the deps.", "3.1. Install the deps."),
            ("2.1) Install the deps.", "2.4) Install the deps."),
            ("(1) Install the deps.", "(2) Install the deps."),
            ("(a) Install the deps.", "(b) Install the deps."),
            ("a. Install the deps.", "b. Install the deps."),
            ("A) Install the deps.", "C) Install the deps."),
            ("i. Install the deps.", "ii. Install the deps."),
            ("(iv) Install the deps.", "(vi) Install the deps."),
            ("IX. Install the deps.", "XI. Install the deps."),
            ("Phase 1: Install the deps.", "Phase 2: Install the deps."),
            ("Stage 3 - Install the deps.", "Stage 4 - Install the deps."),
            ("**Part 1.** Install the deps.", "**Part 2.** Install the deps."),
            ("### 2.1 Install the deps", "### 3.2 Install the deps"),
            ("第1步：安装依赖", "第2步：安装依赖"),
            ("第一步，安装依赖", "第二步，安装依赖"),
            ("- 第3阶段: 安装依赖", "- 第4阶段: 安装依赖"),
        ):
            assert added_words(before, after) == (0, 0), (before, after)
        # A number that is content stays a word: a quantity, a version, a
        # count, a phase named inside a sentence or with no separator, a
        # step of a Chinese sentence.
        for before, after in (
            ("2.5 kg of flour.", "3.5 kg of flour."),
            ("Phase 2 of the rollout fails.", "Phase 3 of the rollout fails."),
            ("Retry phase 1 twice.", "Retry phase 2 twice."),
            ("Use v2.1 of the API.", "Use v2.2 of the API."),
            ("3 retries are enough.", "5 retries are enough."),
            ("第1步的输出", "第2步的输出"),
        ):
            assert added_words(before, after) == (1, 1), (before, after)
        # Words that look like a marker only at the head of a list item.
        assert added_words("Mix. Then stir.", "Stir. Then mix.") == (2, 2)

    def test_a_reordering_is_counted_in_linear_time(self):
        """The reorder count must stay linear: the diff runs inline on the
        event loop (a super-linear word diff would block it)."""
        words = [f"w{i % 37}" for i in range(24_000)]
        shuffled = words[1:] + words[:1]
        cases = [
            # One line each, the same bag: a phrase loop reordered.
            ("check the value. " * 3000, "value the check. " * 3000),
            # A region over the line-alignment cap: one block of 24k words.
            (
                "\n".join(" ".join(words[i : i + 7]) for i in range(0, len(words), 7)),
                "\n".join(
                    " ".join(shuffled[i : i + 5]) for i in range(0, len(words), 5)
                ),
            ),
            # A region under the cap whose lines all changed (each reversed).
            (
                "\n".join(" ".join(words[i : i + 120]) for i in range(0, 24_000, 120)),
                "\n".join(
                    " ".join(reversed(words[i : i + 120]))
                    for i in range(0, 24_000, 120)
                ),
            ),
        ]
        for old, new in cases:
            start = time.perf_counter()
            added, removed = added_words(old, new)
            elapsed = time.perf_counter() - start
            assert elapsed < 1.0, (old[:20], new[:20], elapsed)
            assert added > 0 and added == removed, (old[:20], added, removed)


class TestDroppedSentences:
    def test_a_rewrite_keeps_verbatim_and_reworded_sentences(self):
        old = (
            "Look in every drawer before giving up. "
            "Report the final location of every object you moved. "
            "Never treat a cup as a mug in any task."
        )
        new = "Look in every drawer before giving up. Report the final location of each object that you moved."
        dropped, total = dropped_sentences(old, new)
        assert total == 3 and dropped == ["Never treat a cup as a mug in any task."]
        assert dropped_sentences(old, old) == ([], 3)
        # Fragments under four words are not counted.
        assert dropped_sentences("Open it. Close it.", "") == ([], 0)

    def test_a_chinese_rewrite_reports_its_dropped_sentence(self):
        """A CJK sentence has no spaces: it was one "word", under the
        four-word minimum, so a Chinese rewrite never got the report."""
        old = "先检查配置文件是否正确。然后部署到生产环境。最后验证服务是否正常运行。"
        new = "先检查配置文件是否正确。最后验证服务是否正常运行。"
        assert dropped_sentences(old, new) == (["然后部署到生产环境。"], 3)
        assert dropped_sentences(old, old) == ([], 3)


class TestNumbering:
    def test_repeated_and_skipped_numbers(self):
        text = "1. a\n2. b\n3. c\n10. d\n4. e\n6. f\n6. g"
        assert numbering_problems(text) == ["3→10", "10→4", "4→6", "6 repeated"]

    def test_valid_lists(self):
        assert numbering_problems("1. a\n2. b\n3. c") == []
        assert numbering_problems("1. a\n2. b\nprose\n1. c\n2. d") == []
        # A nested list is numbered on its own; decimals are not list items.
        assert numbering_problems("1. a\n   1. x\n   2. y\n2. b\n3.5 kg of flour") == []


class TestCitationQuoteProblem:
    """A citation's quote must be text the write added."""

    def test_found_in_the_new_text_and_not_in_the_old(self):
        from mirix.services.skill_text_checks import citation_quote_problem

        assert citation_quote_problem(
            "Weigh every parcel", "Open it. Weigh every parcel first.", "Open it."
        ) is None
        # Spacing, case, punctuation and line breaks do not matter.
        assert citation_quote_problem(
            "weigh  EVERY\nparcel!", "Weigh every parcel first.", ""
        ) is None

    def test_the_reasons(self):
        from mirix.services.skill_text_checks import (
            QUOTE_MISSING,
            QUOTE_NOT_FOUND,
            QUOTE_TOO_SHORT,
            QUOTE_UNCHANGED,
            citation_quote_problem,
        )

        assert citation_quote_problem(None, "x") == QUOTE_MISSING
        assert citation_quote_problem("   ", "x") == QUOTE_MISSING
        assert citation_quote_problem("the box", "open the box") == QUOTE_TOO_SHORT
        assert (
            citation_quote_problem("label every parcel", "Weigh every parcel.")
            == QUOTE_NOT_FOUND
        )
        assert (
            citation_quote_problem(
                "weigh every parcel", "Weigh every parcel.", "Weigh every parcel."
            )
            == QUOTE_UNCHANGED
        )

    def test_an_elided_quote_needs_every_part_and_one_new_part(self):
        from mirix.services.skill_text_checks import (
            QUOTE_NOT_FOUND,
            QUOTE_UNCHANGED,
            citation_quote_problem,
        )

        new = "Weigh every parcel. Then record the weight on the form."
        assert citation_quote_problem("Weigh every parcel ... record the weight", new) is None
        assert (
            citation_quote_problem("Weigh every parcel … stamp the form", new)
            == QUOTE_NOT_FOUND
        )
        assert (
            citation_quote_problem(
                "Weigh every parcel ... the form", new, "Weigh every parcel."
            )
            == QUOTE_UNCHANGED
        )

    def test_cjk_characters_weigh_more(self):
        from mirix.services.skill_text_checks import citation_quote_problem

        assert citation_quote_problem("先称重再贴标签", "先称重再贴标签，然后封箱。") is None


class TestQuotesKeepCombiningMarks:
    """A script that writes its vowels as combining marks (Devanagari) keeps
    them when a quote is squashed: they count toward the minimum and must
    match."""

    def test_an_indic_quote_counts_whole(self):
        from mirix.services.skill_text_checks import (
            citation_quote_problem,
            squash_quote_text,
        )

        quote = "पहले लेबल जांचें"
        assert len(squash_quote_text(quote)) == len(quote.replace(" ", ""))
        new = "डिब्बा खोलें। पहले लेबल जांचें, फिर भेजें।"
        assert citation_quote_problem(quote, new) is None
        # Only the consonants of it do not match any more.
        assert citation_quote_problem("पहल लबल जच", new) is not None

    def test_ascii_text_squashes_as_before(self):
        from mirix.services.skill_text_checks import squash_quote_text

        assert squash_quote_text("Check  the_Label, first!") == "checkthelabelfirst"
        assert squash_quote_text("Ｃａｆé ①") == "café1"
