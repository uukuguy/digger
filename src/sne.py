#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import OrderedDict
#from bokeh.sampledata.iris import flowers
from bokeh.plotting import *
from bokeh.charts import Scatter
import pandas
from pandas import DataFrame
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sample_feature_matrix import SampleFeatureMatrix
from category_feature_matrix import CategoryFeatureMatrix
from categories import Categories

def sne(samples, result_dir, include_null_samples):
    csv_file = "%s/%s_sne.csv" % (result_dir, samples.name)
    html_file = "%s/%s.html" % (result_dir, samples.name)
    title = 'Negative Opinions'

    cfm, sfm = samples.get_categories_1_weight_matrix()
    sfm.save_to_svmfile("./result/%s_sfm.svm" % (samples.name), include_null_samples)

    X, y = sfm.to_sklearn_data(include_null_samples)
    total_categories = sfm.get_num_categories()

    logging.debug("Do TSNE...")
    model = TSNE(n_components=2, random_state=0)
    X_sne = model.fit_transform(X)

    logging.debug("Saving %s..." % (csv_file))
    f = open(csv_file, "wb+")
    f.write("x,y,cat,t\n")
    rowidx = 0
    categories = samples.get_categories()
    for row in X_sne:
        for col in row:
            f.write("%.3f," % (col))
        category_idx = y[rowidx]
        category_id = sfm.get_category_id(category_idx)
        category_1_id = Categories.get_category_1_id(category_id)
        category_name = "%s(%d)" % ((~categories.categories_1)[category_1_id], category_1_id)
        f.write("%s,%d\n" % (category_name.encode('utf-8'), int(category_1_id / 1000000)))
        rowidx += 1
    f.close()

    logging.info("SNE CSV file %s saved." % (csv_file))
    #plt_show(X_sne, y, total_categories)
    show_diagram(csv_file, html_file, title)


def plt_show(data, targets, total_categories):

    plt.figure(figsize=(12,8))
    plt.title("SNE")
    #plt.barh(indices, score, .2, label="score", color='r')
    #plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    #plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')

    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)


    #[color_map] = np.random.random((1, total_categories))
    color_map = [
    "#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]

    print color_map
    x = [ r[0] for r in data]
    y = [ r[1] for r in data]
    c = [ color_map[t] for t in targets]

    plt.scatter(x, y, c=c, s=3)

    plt.show()

def show_diagram(csv_file, html_file, title):
    df = pandas.read_csv(csv_file)
    colormap = {
        0:"#fbdf6f", 1:"#b2dfca", 2:"#fb9a99", 3:"#1f78b4", 4:"#14ad80", 5:"#a6ce43",
        6:"#e31a1c", 7:"#00ffff", 8:"#6a3d9a", 9:"#808080", 10:"#cab2d6"}
    df['color'] = df['t'].map(lambda x: colormap[x])

    output_file(html_file, title=title)

    p = figure(title = title, tools="resize,crosshair,pan,wheel_zoom,box_zoom,reset,previewsave")
    p.xaxis.axis_label = 'X'
    p.yaxis.axis_label = 'Y'

    values = df.groupby("cat")
    pdict = OrderedDict()
    for cat in values.groups:
        print cat
        labels = values.get_group(cat).columns
        print labels
        xname = labels[0]
        yname = labels[1]
        colorname = labels[4]
        print xname, yname
        x = getattr(values.get_group(cat), xname)
        y = getattr(values.get_group(cat), yname)
        item_color = getattr(values.get_group(cat), colorname)
        pdict[cat] = zip(x,y)
        print type(pdict[cat])
        #print pdict[cat]
        for (x,y) in pdict[cat]:
            print x, y
            p.circle(x, y, legend=cat, color=item_color, fill_alpha = 0.2, size=5)
            break
            ##p.scatter(x, y, radius=20, color=item_color, legend=cat, fill_alpha=0.2)

    p.circle(df["x"], df["y"], color=df["color"], fill_alpha=0.2, size=5)

    show(p)

if __name__ == '__main__':
    csv_file = 'result/no2013_sne.csv'
    html_file = 'no2013.html'
    title = 'Negative Opinions'
    show_diagram(csv_file, html_file, title)
