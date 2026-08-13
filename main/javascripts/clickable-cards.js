// Make each `grid cards` <li> fully clickable on pages that opt in via
// a hidden `<div class="tutorial-cards">` marker. Clicking anywhere on
// the card navigates to the card's last <a> (the arrow link); clicks
// on real links, text selections, and modifier-clicks are left alone.
(function () {
  function activate() {
    if (!document.querySelector(".tutorial-cards")) return;
    const cards = document.querySelectorAll(".grid.cards > ul > li");
    cards.forEach((li) => {
      const links = li.querySelectorAll("a[href]");
      const target = links[links.length - 1];
      if (!target) return;
      li.style.cursor = "pointer";
      li.addEventListener("click", (event) => {
        if (event.target.closest("a")) return;
        if (window.getSelection && window.getSelection().toString()) return;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1) {
          window.open(target.href, "_blank");
          return;
        }
        window.location.href = target.href;
      });
    });
  }

  // Material for MkDocs uses instant navigation; re-run on each page change.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(activate);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", activate);
  } else {
    activate();
  }
})();
