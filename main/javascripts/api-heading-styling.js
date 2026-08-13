// Enhance API reference headings with a visual hierarchy for the fully
// qualified name.
//
// mkdocstrings-python renders each class/function/attribute heading in
// one of two forms depending on the symbol:
//
//   A. Signature run through Pygments (functions, classes with __init__,
//      attributes) — leading dotted name appears as a sequence of `.n`
//      identifiers separated by `.o` dots inside a `<code class="highlight">`:
//
//        <h3 class="doc doc-heading">
//          <code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>
//          <code class="highlight">
//            <span class="n">fastforward</span><span class="o">.</span>
//            <span class="n">mpath</span><span class="o">.</span>
//            <span class="n">MyClass</span><span class="p">(</span>...
//          </code>
//        </h3>
//
//   B. Plain-text fallback (classes without an explicit `__init__` — e.g.
//      enums, dataclasses, subclasses using inherited `__init__`) — the
//      whole name is emitted as raw text inside a plain `<code>`:
//
//        <h3 class="doc doc-heading">
//          <code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>
//          <code>fastforward.dispatcher.DispatcherPriority</code>
//        </h3>
//
// In both cases, the goal is to mark the last dotted segment with
// `.doc-heading-name` (styled in the primary accent by extra.css) and
// everything before it with `.doc-heading-qualifier` (muted). Un-bolding
// and literal-token coloring is handled in docs/stylesheets/extra.css.

document.addEventListener("DOMContentLoaded", () => {
  const headings = document.querySelectorAll(".doc-heading");
  headings.forEach((heading) => {
    for (const code of heading.querySelectorAll("code")) {
      if (code.classList.contains("doc-symbol")) continue;
      if (code.querySelector("span.n")) {
        classifyPygmentsLeadingName(code);
      } else if (code.textContent.trim()) {
        classifyPlainDottedName(code);
      }
      return;
    }
  });
});

function classifyPygmentsLeadingName(code) {
  const leading = [];
  let expectingIdent = true;
  for (const child of Array.from(code.children)) {
    if (expectingIdent) {
      if (child.classList.contains("n")) {
        leading.push(child);
        expectingIdent = false;
      } else {
        break;
      }
    } else {
      if (child.classList.contains("o") && child.textContent === ".") {
        leading.push(child);
        expectingIdent = true;
      } else {
        break;
      }
    }
  }
  if (leading.length === 0) return;

  const lastIdent = leading[leading.length - 1];
  lastIdent.classList.add("doc-heading-name");
  for (let i = 0; i < leading.length - 1; i++) {
    leading[i].classList.add("doc-heading-qualifier");
  }
}

function classifyPlainDottedName(code) {
  const text = code.textContent;
  const dotIdx = text.lastIndexOf(".");

  code.textContent = "";

  if (dotIdx >= 0) {
    const qualifier = document.createElement("span");
    qualifier.className = "doc-heading-qualifier";
    qualifier.textContent = text.slice(0, dotIdx + 1);
    code.appendChild(qualifier);
  }

  const name = document.createElement("span");
  name.className = "doc-heading-name";
  name.textContent = text.slice(dotIdx + 1);
  code.appendChild(name);
}
