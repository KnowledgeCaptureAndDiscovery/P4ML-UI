# dsbox-corex
Corex primitive running versions of the algorithm for discrete (using "bio_corex"), continuous (using "linear_corex": fastest version, gaussian), and text data (using "corex_topic", analogous to LDA).

ARGUMENTS:

-Number of latent factors fed as individual disc, cont, and text arguments, or as a latent_pct for the number of discrete/continuous columns.  

-Columns to be analysed can be fed as a dictionary of lists of strings to the separate_types argument (with keys 'disc', 'cont', 'text'), Can also be read from given data_schema if all columns are to be analysed.  (TO DO: change separate_types to individual list arguments for consistency with factors or vice versa?)

-Text preprocessing into binary bag of words handled internally rather than via separate primitive (featurize_text = True)

-Also supports reading from raw text files with argument data_path (assuming column consists of file_names)

