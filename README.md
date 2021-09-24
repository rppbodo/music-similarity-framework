# music-similarity-framework

## dependencies

this framework was developed and tested using [anaconda](https://www.anaconda.com/).

you can clone our environment using the file [environment.yml](https://github.com/rppbodo/music-similarity-framework/blob/main/environment.yml).

also, it is necessary to manually install [https://github.com/tuwien-musicir/rp_extract](https://github.com/tuwien-musicir/rp_extract)

## dataset format

this framework expects the dataset tree structure to be like this:

![tree_structure.jpg](https://github.com/rppbodo/music-similarity-framework/blob/main/img/tree_structure.jpg)

## file list

this framework expects the file /path/to/dataset/tracks.csv with the following format:

"class_1","file_1"
...
"class_1","file_N1"
"class_2","file_1"
...
"class_2","file_N2"
...
"class_M","file_1"
...
"class_M","file_NM"

## experimented datasets

this framework was executed with the following datasets:

* [Ballroom](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)
* [GTZAN](http://marsyas.info/downloads/datasets.html)
* [IOACAS-QBH](http://mirlab.org/dataSet/public/IOACAS_QBH.rar)
* [Maria Panteli’s melody dataset](https://archive.org/details/panteli_maria_melody_dataset)
* [Maria Panteli’s rhythm dataset](https://archive.org/details/panteli_maria_rhythm_dataset)
* [MAST](https://zenodo.org/record/2620357)
* [1517-Artists](http://www.seyerlehner.info/index.php?p=1_3_Download)
* [MIR-QBSH](http://mirlab.org/dataSet/public/MIR-QBSH-corpus.rar)
* [FMA-Small](https://github.com/mdeff/fma/)

the results can be found at [this link](https://rppbodo.github.io/phd/experiment_1.html)

---

also with datasets specifically designed for the cover song identification problem:

* [Covers80](https://labrosa.ee.columbia.edu/projects/coversongs/covers80/)
* [YouTubeCovers](https://sites.google.com/site/ismir2015shapelets/data)
* [Covers1000](http://www.covers1000.net/)
* [Mazurkas](http://www.mazurka.org.uk/)
* [SHS9K](https://rppbodo.github.io/phd/shs9k.html)

the results can be found at [this link](https://rppbodo.github.io/phd/experiment_2.html)

