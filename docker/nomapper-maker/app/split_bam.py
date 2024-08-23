#!/usr/bin/python3

"""split_bam.py
Script to split bam file due to flags.
"""

import pysam
import argparse


def split(filename):
    output_dir = '/vol'

    bamfile = pysam.AlignmentFile(filename, 'rb')
    flag_mapped = pysam.AlignmentFile(f"{output_dir}/mapped.bam", "wb", template=bamfile)
    flag_unmapped = pysam.AlignmentFile(f"{output_dir}/unmapped.bam", "wb", template=bamfile)

    for line in bamfile:
        flag = line.flag
        if flag == 4:
            flag_unmapped.write(line)
        elif flag == 0 or flag == 16:
            flag_mapped.write(line)

    bamfile.close()
    flag_mapped.close()
    flag_unmapped.close()

def main():
    parser = argparse.ArgumentParser(description='Script to split bam file due to flags.')
    parser.add_argument('-f', '--file', type=str, default='/vol/data.bam', help='input file path (.bam)')
    args = parser.parse_args()
    
    split(args.file)


if __name__ == '__main__':
    main()