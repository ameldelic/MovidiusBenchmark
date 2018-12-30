import csv
import os

def clear_benchmark():
    os.remove('benchmark.csv')

def enter_benchmark(target, net, framework, mode, time):
    bench = [target, net, framework, mode, time]

    with open('benchmark.csv', mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(bench)
