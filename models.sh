#!/bin/bash

cat models.csv | while read line; do
	extractor=$(echo ${line} | cut -d ',' -f 1)
	aggregator=$(echo ${line} | cut -d ',' -f 2)
	distance=$(echo ${line} | cut -d ',' -f 3)
	
	python 1_extract.py ${1} ${extractor}
	python 2_aggregate.py ${1} ${extractor} ${aggregator}
	python 3_distances.py ${1} ${extractor} ${aggregator} ${distance}
	python 4_metrics.py ${1} ${extractor} ${aggregator} ${distance} --similarity
	python 4_metrics.py ${1} ${extractor} ${aggregator} ${distance} --cover
	exit
done

