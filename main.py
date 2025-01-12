from gui import run_gui
from common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="""
            Path to model used for text information extraction using spaCy.
            """)
    parser.add_argument('-t', '--text_path', type=str, help="""
            Path to file containing text to summarize.
            """)
    parser.add_argument('-s', '--summary_methods', type=str, help="""
            Methods used, available
            options: EXT_NAIVE|EXT_FIRST_LAST|EXT_IF_IDF|EXT_TEXT_RANK or
            ABST_T5|ABST_PEGASUS|ABST_BART. In the case of ABST_ options, order
            determines in what order methods are applied.
            """)
    parser.add_argument('-j', '--json_results_path', type=str, help="""
            Json path to which to save the results.
            """)
    parser.add_argument('-r', '--reference_path', type=str, help="""
            (Optional) Path to file containing reference summary.
            """)
    parser.add_argument('-p', '--percentage', type=str, help="""
                Percentage of sentences in the summary. Integer numbers seperated by |. In the case of ABST_ methods, 
                order determines in what order percentages are applied. E.g. 25|10|5
                """)
    parser.add_argument('-g', '--gui', action='store_true', help="""Enable GUI mode.""")
    args = parser.parse_args()

    if args.gui:
        run_gui()
    else:
        methods = parse_method(args.summary_methods)
        print(methods)

        with open(args.text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        with open(args.reference_path, 'r', encoding='utf-8') as f:
            reference_summary = f.read()

        nlp = spacy.load(args.model)
        nlp.add_pipe("textrank")
        doc = nlp(text)

        if is_method_set(methods, EXT_SUMMARY):
            summary = extraction_based_summarize(doc, methods, int(math.ceil((float(args.percentage.split('|')[0])/100.0)*float(len(list(doc.sents))))))
            print(summary)
            if args.reference_path:
                eval_extraction_based = \
                    evaluate_summary(summary, reference_summary)
                print_evaluation(eval_extraction_based)
                save_result_to_json(args, methods, eval_extraction_based, summary)

        if is_method_set(methods, ABST_SUMMARY):
            methods_list = args.summary_methods.split('|')
            percentages_list = args.percentage.split('|')
            if not args.percentage or len(percentages_list) != len(methods_list):
                raise Exception("No percentages parameter or different methods and percentages sizes!")
            num_of_tokens_list = [int(len(doc) * int(per) / 100) for per in percentages_list]
            abstractive_summary = abstractive_summarization(text, methods_list, num_of_tokens_list, doc, nlp)
            print(abstractive_summary)
            if args.reference_path:
                eval_abstractive = evaluate_summary(abstractive_summary,
                                                    reference_summary)
                print_evaluation(eval_abstractive)
                save_result_to_json(args, methods, eval_abstractive, abstractive_summary)
# ...existing code...
