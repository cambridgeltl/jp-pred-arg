# jp-pred-arg
Japanese selectional preferences and predicate-argument identfication in TensorFlow

# Preparing the Data
## NAIST Text Corpus (NTC)

You can find information for obtaining the NTC here, though it is primarily distributed by CD and admittedly difficult for foreign researchers:
https://sites.google.com/site/naisttextcorpus/

Assuming you're able to obtain the data, generate the established (in Ouchi et al. 2015, 2016) data splits with the following commands:

Train:

`cat data/NaistTextCorpus/NTC_1.5/dat/ntc/knp/95010[1-9]* > train.ntc`

`cat data/NaistTextCorpus/NTC_1.5/dat/ntc/knp/9501[10,11]* >> train.ntc`

`cat data/NaistTextCorpus/NTC_1.5/dat/ntc/knp/950[1-8]ED* >> train.ntc`

Dev:

`cat data/NaistTextCorpus/NTC_1.5/dat/ntc/knp/9501[12,13]* > dev.ntc`

`cat data/NaistTextCorpus/NTC_1.5/dat/ntc/knp/9509ED* >> dev.ntc`

Test:

`cat data/NaistTextCorpus/NTC_1.5/dat/ntc/knp/9501[14,15,16,17]* > test.ntc`

`cat data/NaistTextCorpus/NTC_1.5/dat/ntc/knp/951[0-2]ED* >> test.ntc`
