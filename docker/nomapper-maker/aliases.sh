alias trim="porechop -i /vol/input.fastq.gz --format 'fastq' --barcode_threshold '75.0' --barcode_diff '5.0' --adapter_threshold '90.0' --check_reads '10000' --scoring_scheme '3,-6,-5,-2' --end_size '150' --min_trim_size '4' --extra_end_trim '2' --end_threshold '75.0' --middle_threshold '85.0' --extra_middle_trim_good_side '10' --extra_middle_trim_bad_side '100' --min_split_read_size '1000' -o /vol/data_porechop.fastq"
alias filter="filtlong --min_length '500' --length_weight '1.0' --mean_q_weight '1.0' --window_q_weight '1.0' --window_size '250' /vol/data_porechop.fastq > /vol/data_filtlong.fastq"
alias map="minimap2 -x splice -k 14 --q-occ-frac 0.01 --frag=no --splice -u f --splice-flank=yes -L --cs=long -t 8 -a '/vol/ref.fna' '/vol/data_filtlong.fastq' > '/vol/data.sam'"
alias sam2bam="samtools view -bS '/vol/data.sam' > '/vol/data.bam'"

alias split="python3 split_bam.py"
alias train="python3 train.py && chmod 777 /vol/model.h5"

alias preprocess="trim && filter && map && sam2bam && split"
alias do-all="trim && filter && map && sam2bam && split && train"

alias clean-all="cd /vol/ && rm data_porechop.fastq data_filtlong.fastq data.sam data.bam mapped.bam unmapped.bam"