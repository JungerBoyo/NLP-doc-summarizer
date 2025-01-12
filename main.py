import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import torch
import math
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox

EXT_NAIVE_SCORING = 0x01
EXT_FIRST_LAST_SCORING = 0x02
EXT_TF_IDF_SCORING = 0x04
EXT_TEXT_RANK_SCORING = 0x08
EXT_SUMMARY = 0x0F

ABST_T5 = 0x10
ABST_PEGASUS = 0x20
ABST_BART = 0x40
ABST_SUMMARY = 0xF0

METHODS_MAP = dict([
    ('EXT_NAIVE', EXT_NAIVE_SCORING),
    ('EXT_FIRST_LAST', EXT_FIRST_LAST_SCORING),
    ('EXT_TF_IDF', EXT_TF_IDF_SCORING),
    ('EXT_TEXT_RANK', EXT_TEXT_RANK_SCORING),
    ('ABST_T5', ABST_T5),
    ('ABST_PEGASUS', ABST_PEGASUS),
    ('ABST_BART', ABST_BART)
])


def is_method_set(method, bit):
    return (method & bit) > 0


def parse_method(method_str):
    methods = method_str.split('|')
    result = 0
    for method_str in methods:
        method = METHODS_MAP.get(method_str)
        if method is None:
            return 0
        result |= method
    return result


def extraction_based_summarize_naive_scoring(doc):
    sentence_scores = {}

    for i, sent in enumerate(doc.sents):
        score = 0
        for token in sent:
            if token.pos_ in ["NOUN", "ADJ", "VERB"]:
                score += 1
        score /= len(sent)
        sentence_scores[i] = score
    return sentence_scores


def extraction_based_summarize_tfidf_sentence_scoring(doc):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([sent.text for sent in doc.sents])
    sentence_scores = X.sum(axis=1)
    max_score = sentence_scores.flatten().max()
    min_score = sentence_scores.flatten().min()
    sorted_indices = sentence_scores.flatten().argsort()

    sent_score_map = {}
    for sent_id in sorted_indices[0, :].tolist()[0]:
        sent_score_map[sent_id] = (sentence_scores[sent_id] - min_score) / \
            (max_score - min_score)

    return sent_score_map


def extraction_based_summarize_text_rank_sentence_scoring(doc):
    doc_sents_list = list(doc.sents)
    sentence_scores = {}

    for j, sent_tr in enumerate(doc._.textrank.summary(
            limit_sentences=len(doc_sents_list))):
        sent_id = 0
        for i, sent in enumerate(doc_sents_list):
            if sent.text == sent_tr.text:
                sent_id = i
                break
        score = float(j) / float(len(doc_sents_list))
        sentence_scores[sent_id] = score

    return sentence_scores


def add_base(sents_map, base):
    for sent_id, score in sents_map.items():
        sents_map[sent_id] = score + base
    return sents_map


def extraction_based_summarize(doc, method, num_sentences):
    doc_sents_list = list(doc.sents)

    use_naive_scoring = is_method_set(method, EXT_NAIVE_SCORING)
    use_first_last_scoring = is_method_set(method, EXT_FIRST_LAST_SCORING)
    use_tf_idf_scoring = is_method_set(method, EXT_TF_IDF_SCORING)
    use_text_rank_scoring = is_method_set(method, EXT_TEXT_RANK_SCORING)

    first_sent = doc_sents_list[0]
    last_sent = doc_sents_list[-1]
    if use_first_last_scoring and num_sentences == 1:
        return first_sent
    if use_first_last_scoring and num_sentences == 2:
        return str(first_sent).strip() + " " + str(last_sent).strip()

    if use_first_last_scoring:
        num_sentences -= 2

    sent_id_score_maps = []

    if use_naive_scoring:
        naive_sents_map = extraction_based_summarize_naive_scoring(doc)
        sent_id_score_maps.append(add_base(naive_sents_map, 0.2))

    if use_tf_idf_scoring:
        tfidf_sents_map = \
                extraction_based_summarize_tfidf_sentence_scoring(doc)
        sent_id_score_maps.append(add_base(tfidf_sents_map, 0.4))

    if use_text_rank_scoring:
        text_rank_sents_map = \
                extraction_based_summarize_text_rank_sentence_scoring(doc)
        sent_id_score_maps.append(add_base(text_rank_sents_map, 0.7))

    final_map = {}
    for sent_map in sent_id_score_maps:
        for sent_id, score in sent_map.items():
            if sent_id not in final_map:
                final_map[sent_id] = score
            else:
                final_map[sent_id] += score

    summary_sentences = []
    sorted_final_map_items = sorted(final_map.items(), key=lambda x: x[1],
                                    reverse=True)
    for sent_id, score in sorted_final_map_items:
        summary_sentences.append((sent_id, doc_sents_list[sent_id]))
    summary_sentences = summary_sentences[:num_sentences]
    summary_sentences = sorted(summary_sentences, key=lambda x: x[0],
                               reverse=True)
    summary = " ".join([str(sent).strip() for _, sent in summary_sentences])

    if use_first_last_scoring:
        return str(first_sent).strip() + " " + summary + " " + \
                    str(last_sent).strip()
    else:
        return summary

def generate_abstract_summary(tokenizer, model, num_of_tokens, doc, device, addPrefix=False):
    max_chunk_length = 512
    chunks = [doc[i:i + max_chunk_length].text for i in range(0, len(doc), max_chunk_length)]
    length_per_chunk = int(num_of_tokens / len(chunks))
    if addPrefix:
        prefix = "summarize: "
        chunks = [prefix + doc for doc in chunks]

    summaries = []
    for chunk in chunks:
        batch = tokenizer(chunk, max_length=max_chunk_length, padding=True, truncation=True, return_tensors="pt").to(
            device)
        translated = model.generate(min_length=length_per_chunk, max_length=length_per_chunk, **batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        summaries.append(tgt_text[0])
    return " ".join(summaries)

def abstractive_summarization(text, methods_list, num_of_tokens_list, doc, nlp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = doc

    for idx in range(len(methods_list)):
        method = parse_method(methods_list[idx])
        use_t5 = is_method_set(method, ABST_T5)
        use_pegasus = is_method_set(method, ABST_PEGASUS)
        use_bart = is_method_set(method, ABST_BART)
        num_of_tokens = num_of_tokens_list[idx]

        if use_pegasus:
            model_name = "google/pegasus-xsum"
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
            summary = nlp(generate_abstract_summary(tokenizer, model, num_of_tokens, summary, device))

        if use_t5:
            model_name = "t5-base"
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            summary = nlp(generate_abstract_summary(tokenizer, model, num_of_tokens, summary, device, True))

        if use_bart:
            model_name = "facebook/bart-large-cnn"
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            summary = nlp(generate_abstract_summary(tokenizer, model, num_of_tokens, summary, device))
        print(summary)

    return summary.text

def evaluate_summary(generated_summary, reference_summary):
    """
    Ocena jakości podsumowania za pomocą metryk ROUGE i BERTScore.
    """
    # ROUGE
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                     use_stemmer=True)
    rouge_scores = rouge.score(reference_summary, generated_summary)

    # BERTScore
    P, R, F1 = bert_score([generated_summary], [reference_summary], lang="en")

    evaluation = {
        'ROUGE': {k: v.fmeasure for k, v in rouge_scores.items()},
        'BERTScore': {
            'Precision': P.mean().item(),
            'Recall': R.mean().item(),
            'F1': F1.mean().item()
        }
    }
    return evaluation

def print_evaluation(evaluation):
    for metric, pairs in evaluation.items():
        print(f'{metric}:')
        for key, value in pairs.items():
            print(f'\t{key} = {value}')
        print()

def add_value_to_json_array(file_path, key, new_value):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}

    if key in data:
        if isinstance(data[key], list):
            data[key].append(new_value)
    else:
        data[key] = [new_value]

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def save_result_to_json(args, methods, evaluation, summary):
    evaluation['summary'] = summary
    evaluation['methods'] = args.summary_methods
    base_text_path = os.path.basename(args.text_path)

    add_value_to_json_array(args.json_results_path, base_text_path, evaluation)

def remove_abstractive_method():
    summary_type = summary_type_var.get()
    if summary_type == "extractive":
        add_extractive_method()
    else:
        add_abstractive_method()

def remove_method():
    summary_type = summary_type_var.get()
    if summary_type == "extractive":
        remove_extractive_method()
    else:
        remove_abstractive_method()

def add_extractive_method():
    method = method_var.get()
    if not method:
        messagebox.showerror("Error", "Please fill in both method and percentage fields.")
        return
    methods_listbox.insert(tk.END, method)

def remove_extractive_method():
    selected_indices = methods_listbox.curselection()
    for index in selected_indices[::-1]:
        methods_listbox.delete(index)

def update_method_options():
    summary_type = summary_type_var.get()
    if summary_type == "extractive":
        method_options = ["EXT_NAIVE", "EXT_FIRST_LAST", "EXT_TF_IDF", "EXT_TEXT_RANK"]
        methods_listbox.delete(0, tk.END)
    else:
        method_options = ["ABST_T5", "ABST_PEGASUS", "ABST_BART"]
        methods_listbox.delete(0, tk.END)
    method_var.set(method_options[0])
    method_menu['menu'].delete(0, 'end')
    for option in method_options:
        method_menu['menu'].add_command(label=option, command=tk._setit(method_var, option))

def add_abstractive_method():
    method = method_var.get()
    percentage = percentage_entry.get()
    if not method or not percentage:
        messagebox.showerror("Error", "Please fill in both method and percentage fields.")
        return
    methods_listbox.insert(tk.END, f"{method}|{percentage}")
    percentage_entry.delete(0, tk.END)

def remove_abstractive_method():
    selected_indices = methods_listbox.curselection()
    for index in selected_indices[::-1]:
        methods_listbox.delete(index)

def browse_json_directory():
    directory = filedialog.askdirectory()
    if directory:
        json_directory_entry.delete(0, tk.END)
        json_directory_entry.insert(0, directory)

def browse_reference_summary():
    file_path = filedialog.askopenfilename()
    if file_path:
        reference_summary_entry.delete(0, tk.END)
        reference_summary_entry.insert(0, file_path)

def summarize():
    text_path = text_path_entry.get()
    summary_type = summary_type_var.get()
    reference_summary_path = reference_summary_entry.get()
    json_directory = json_directory_entry.get()
    json_filename = json_filename_entry.get()

    if not text_path or not summary_type:
        messagebox.showerror("Error", "Please fill in all required fields.")
        return

    if summary_type == "extractive":
        methods = "|".join([item for item in methods_listbox.get(0, tk.END)])
        percentage = percentage_entry.get()
        if not methods:
            messagebox.showerror("Error", "Please add at least one method.")
            return
        percentages = percentage
    else:
        methods = "|".join([item.split("|")[0] for item in methods_listbox.get(0, tk.END)])
        percentages = "|".join([item.split("|")[1] for item in methods_listbox.get(0, tk.END)])
        if not methods or not percentages:
            messagebox.showerror("Error", "Please add at least one method and percentage.")
            return

    json_output_path = os.path.join(json_directory, json_filename) if json_directory and json_filename else None

    args = argparse.Namespace(
        model="en_core_web_sm",
        text_path=text_path,
        summary_methods=methods,
        json_results_path=json_output_path,
        reference_path=reference_summary_path,
        percentage=percentages
    )

    with open(args.text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    nlp = spacy.load(args.model)
    nlp.add_pipe("textrank")
    doc = nlp(text)

    methods = parse_method(args.summary_methods)
    if is_method_set(methods, EXT_SUMMARY):
        summary = extraction_based_summarize(doc, methods, int(math.ceil((float(args.percentage)/100.0)*float(len(list(doc.sents))))))
    elif is_method_set(methods, ABST_SUMMARY):
        methods_list = args.summary_methods.split('|')
        percentages_list = args.percentage.split('|')
        num_of_tokens_list = [int(len(doc) * int(per) / 100) for per in percentages_list]
        summary = abstractive_summarization(text, methods_list, num_of_tokens_list, doc, nlp)
    else:
        messagebox.showerror("Error", "Invalid summarization method.")
        return

    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, summary)

    if args.reference_path:
        with open(args.reference_path, 'r', encoding='utf-8') as f:
            reference_summary = f.read()
        evaluation = evaluate_summary(summary, reference_summary)
        print_evaluation(evaluation)
        if json_output_path:
            save_result_to_json(args, methods, evaluation, summary)

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
        root = tk.Tk()
        root.title("NLP Document Summarizer")

        tk.Label(root, text="Path to text file:").grid(row=0, column=0, sticky=tk.W)
        text_path_entry = tk.Entry(root, width=50)
        text_path_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=lambda: text_path_entry.insert(0, filedialog.askopenfilename())).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(root, text="Summary type:").grid(row=1, column=0, sticky=tk.W)
        summary_type_var = tk.StringVar(value="extractive")
        tk.Radiobutton(root, text="Extractive", variable=summary_type_var, value="extractive", command=lambda: [update_method_options(), abstractive_frame.grid_remove()]).grid(row=1, column=1, sticky=tk.W)
        tk.Radiobutton(root, text="Abstractive", variable=summary_type_var, value="abstractive", command=lambda: [update_method_options(), abstractive_frame.grid()]).grid(row=1, column=2, sticky=tk.W)

        tk.Label(root, text="Method:").grid(row=2, column=0, sticky=tk.W)
        method_var = tk.StringVar(value="EXT_NAIVE")
        method_menu = tk.OptionMenu(root, method_var, "EXT_NAIVE", "EXT_FIRST_LAST", "EXT_TF_IDF", "EXT_TEXT_RANK")
        method_menu.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(root, text="Percentage:").grid(row=3, column=0, sticky=tk.W)
        percentage_entry = tk.Entry(root, width=50)
        percentage_entry.grid(row=3, column=1, padx=5, pady=5)

        abstractive_frame = tk.Frame(root)
        abstractive_frame.grid(row=4, column=0, columnspan=3, pady=10)
        abstractive_frame.grid_remove()

        tk.Button(abstractive_frame, text="Add Method", command=add_method).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(abstractive_frame, text="Remove Selected Method", command=remove_method).grid(row=0, column=1, padx=5, pady=5)
        methods_listbox = tk.Listbox(abstractive_frame, width=50, height=5)
        methods_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        extractive_frame = tk.Frame(root)
        extractive_frame.grid(row=4, column=0, columnspan=3, pady=10)

        tk.Button(extractive_frame, text="Add Method", command=add_method).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(extractive_frame, text="Remove Selected Method", command=remove_method).grid(row=0, column=1, padx=5, pady=5)
        methods_listbox = tk.Listbox(extractive_frame, width=50, height=5)
        methods_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        tk.Label(root, text="Path to reference summary (optional):").grid(row=5, column=0, sticky=tk.W)
        reference_summary_entry = tk.Entry(root, width=50)
        reference_summary_entry.grid(row=5, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=browse_reference_summary).grid(row=5, column=2, padx=5, pady=5)

        tk.Label(root, text="JSON output directory (optional):").grid(row=6, column=0, sticky=tk.W)
        json_directory_entry = tk.Entry(root, width=50)
        json_directory_entry.grid(row=6, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=browse_json_directory).grid(row=6, column=2, padx=5, pady=5)

        tk.Label(root, text="JSON filename (optional):").grid(row=7, column=0, sticky=tk.W)
        json_filename_entry = tk.Entry(root, width=50)
        json_filename_entry.grid(row=7, column=1, padx=5, pady=5)

        tk.Button(root, text="Summarize", command=summarize).grid(row=8, column=0, columnspan=3, pady=10)

        tk.Label(root, text="Summary output:").grid(row=9, column=0, sticky=tk.W)
        output_text = tk.Text(root, height=10, width=70)
        output_text.grid(row=10, column=0, columnspan=3, padx=5, pady=5)

        root.mainloop()
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
