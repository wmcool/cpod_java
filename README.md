# cpod
Real-time Outlier Detection in Data Streams

Paper link: http://vldb.org/pvldb/vol14/p141-tran.pdf

This project can be opened by Netbeans IDE

After build project, it can be run with command

java -jar CPOD.jar --algorithm  cpod  --R  1.9 --W  10000 --k 50 --slide 500 --datafile  tao.txt --numberWindow  100 --samplingTime 100 

with 10000 is the window size, 500 is slide size, tao.txt is the path to data file, 100 is the number of sliding windows, and the memory and CPU time is sampled after every 100 miliseconds. 

