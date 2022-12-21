import json


def state_of_the_art():
    l = ["mf_avg", "mf_seq", "audio_avg", "audio_seq", "kg_avg", "kg_seq", "hybrid_avg", "hybrid_seq"]

    s = """
\\begin{table}[htbp]
\\centering
\\begin{tabular}{ccccc}
\t\\hline
\t& \\textit{FH@1} & \\textit{FH@5} & \\textit{MRR} & \\textit{MAP@5} \\\\\\hline"""

    for alg in l:
        metrics = json.load(open(f"res/p/theme_prediction/mpd/results/{alg}/best_run/metrics.json"))
        s += "\n\t\\" + alg.replace("_", "") + "{}"
        for m in ["FH@1", "FH@5", "MRR", "MAP@5"]:
            s += " & " + "{:.3f}".format(metrics[m])
        s += " \\\\"

    s += """
\t\\hline
\\end{tabular}
\\caption{}
\\label{}
\\end{table}
"""

    print(s)


def sensitivity_analysis():
    l = ["kg_avg", "kg_int_only_avg", "kg_seq", "kg_int_only_seq"]

    s = """
\\begin{table}[htbp]
\\centering
\\begin{tabular}{ccccc}
\t\\hline
\t& \\textit{FH@1} & \\textit{FH@5} & \\textit{MRR} & \\textit{MAP@5} \\\\\\hline"""

    for alg in l:
        metrics = json.load(open(f"res/p/theme_prediction/mpd/results/{alg}/best_run/metrics.json"))
        s += "\n\t\\" + alg.replace("_", "") + "{}"
        for m in ["FH@1", "FH@5", "MRR", "MAP@5"]:
            s += " & " + "{:.3f}".format(metrics[m])
        s += " \\\\"

    s += """
\t\\hline
\\end{tabular}
\\caption{}
\\label{}
\\end{table}
"""

    print(s)


if __name__ == "__main__":
    state_of_the_art()
    # sensitivity_analysis()
