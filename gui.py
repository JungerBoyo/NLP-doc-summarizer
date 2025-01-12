import tkinter as tk
from tkinter import filedialog, messagebox
from common import *

def run_gui():
    def add_method():
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
