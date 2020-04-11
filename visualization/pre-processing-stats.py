import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def perc_format(pct):
	return "{:.1f}%".format(pct)

def main():
	nazario_stats = 'nazario-analysis.csv'
#	nazario_stats = 'nazario-filter.csv'
	
	df = pd.read_csv(nazario_stats)
	data = df['Count'].tolist()
	labels = df['Category'].tolist()
	
	fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
	wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: perc_format(pct), textprops=dict(color="w"))
	
	ax.legend(wedges, labels, title="Breakdown", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

	plt.setp(autotexts, size=8, weight="bold")

#	ax.set_title("Matplotlib bakery: A pie")

	plt.show()

if __name__== "__main__":
	main()