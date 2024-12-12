#!/bin/bash

text_dir_path_patterns="$1"
number_of_texts=$2
ext_methods=('EXT_NAIVE' 'EXT_FIRST_LAST' 'EXT_IF_IDF' 'EXT_TEXT_RANK')
abst_methods=('ABST_T5' 'ABST_PEGASUS' 'ABST_BART')
t=0

ext_methods_len=$((${#ext_methods[@]} - 1))
echo $ext_methods_len
execute_app() {
	text_path="$1"
	reference_path="$2"
	summary_methods="$3"
	python main.py \
		--model "en_core_web_sm" \
		--text_path "$text_path" \
		--summary_methods "$summary_methods" \
		--num_sentences 5 \
		--reference_path "$reference_path"
}


for t in $(seq 1 $number_of_texts); do
	for _1 in $(seq 0 $ext_methods_len); do
		execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${ext_methods[_1]}"
		for _2 in $(seq $((_1 + 1)) $ext_methods_len); do
			execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${ext_methods[_1]}|${ext_methods[_2]}"
			for _3 in $(seq $(($_2 + 1)) $ext_methods_len); do
				execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${ext_methods[_1]}|${ext_methods[_2]}|${ext_methods[_3]}"
				for _4 in $(seq $(($_3 + 1)) $ext_methods_len); do
					execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${ext_methods[_1]}|${ext_methods[_2]}|${ext_methods[_3]}|${ext_methods[_4]}"
				done
			done
		done
	done
done

#for t in {0..${number_of_texts}}; do
#	for _1 in "${abst_methods[@]}"; do
#		execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_1}"
#		for _2 in "${abst_methods[@]}"; do
#			execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_1}|${_2}"
#			execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_2}|${_1}"
#			for _3 in "${abst_methods[@]}"; do
#				execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_1}|${_2}|${_3}"
#				execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_3}|${_2}|${_1}"
#				execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_2}|${_1}|${_3}"
#				execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_3}|${_1}|${_2}"
#				execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_1}|${_3}|${_2}"
#				execute_app "${text_dir_path_patterns}${t}.txt" "${text_dir_path_patterns}${t}_summary.txt" "${_2}|${_3}|${_1}"
#			done
#		done
#	done
#done
