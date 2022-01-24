(TeX-add-style-hook
 "notes"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "inputenc"
    "fontenc"
    "graphicx"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "amssymb"
    "capt-of"
    "hyperref")
   (LaTeX-add-labels
    "sec:orgce7d909"
    "sec:orge42d7ba"
    "sec:orgfb48cac"
    "sec:orgcc7162a"
    "sec:org8302932"
    "sec:org6e3345a"
    "sec:orga13ec44"
    "sec:org30ebd78"
    "sec:orga6cd5da"
    "sec:orga71c2f8"
    "sec:org4181458"
    "sec:org9f21614"))
 :latex)

