#!/bin/bash

text_dir_path_patterns="$1"
number_of_texts=$2
out="$3"
ext_methods=('EXT_NAIVE' 'EXT_FIRST_LAST' 'EXT_TF_IDF' 'EXT_TEXT_RANK')
abst_methods=('ABST_T5' 'ABST_PEGASUS' 'ABST_BART')
t=0

ext_methods_len=$((${#ext_methods[@]} - 1))
abst_methods_len=$((${#abst_methods[@]} - 1))

execute_app() {
	text_path="$1"
	reference_path="$2"
	summary_methods="$3"
	json_path="$4"
	python main.py \
		--model "en_core_web_sm" \
		--text_path "$text_path" \
		--summary_methods "$summary_methods" \
		--percentage 5 \
		--json_results_path "$json_path" \
		--reference_path "$reference_path"
}

for t in $(seq 1 $number_of_texts); do
	for m in $(seq 0 $ext_methods_len); do
		execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${ext_methods[m]}" "$out"
	done
done

for t in $(seq 1 $number_of_texts); do
	for m in $(seq 0 $abst_methods_len); do
		execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${abst_methods[m]}" "$out"
	done
done
