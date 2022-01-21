(TeX-add-style-hook
 "<none>"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
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
    "hyperref"
    "microtype"
    "xltxtra"
    "fontspec")
   (TeX-add-symbols
    "unifont")
   (LaTeX-add-labels
    "sec:orgfcef277"
    "sec:orgf30ea3e"
    "sec:org152a861"
    "sec:org596cf26"
    "eq:PDE"
    "sec:orgfb92b68"
    "sec:orgcd84ffc"
    "sec:org7a269f1"
    "sec:org88a0c3a"
    "sec:orgdf7e18b"
    "sec:org8c71ae2"
    "sec:org9f4e2ab"
    "sec:orgdaeb947"
    "sec:org652f3cb"
    "sec:org31e6cf1"
    "sec:org17be403"
    "sec:org0826c9c"
    "sec:orge39a504"
    "sec:org4db6895"
    "eq:sixth-order"
    "sec:org826eb72"
    "sec:org6f63cc5"
    "sec:org6bd0046"
    "sec:non-linear"
    "eq:NavEstEQ"
    "sec:org739438d"
    "sec:org4c67a0c"
    "eq:general-2th-order"
    "sec:orgf168fca"
    "sec:num-julia"
    "sec:orgf81e9a7"
    "sec:org7bdb62c"
    "sec:orge93508e"
    "sec:org4224dc3"
    "sec:org6853d7c"
    "sec:org4ea1bf2"
    "sec:org720b5e3"
    "sec:org158d13f"
    "sec:org4073de5"
    "fig:sim1"
    "fig:sim2"
    "sec:orgc197794"
    "sec:orgc48df82"
    "sec:org81e38f9"
    "sec:orgfc290a1"
    "sec:org5215ff6")
   (LaTeX-add-bibliographies
    "../../../Bibliography/collection"))
 :latex)

