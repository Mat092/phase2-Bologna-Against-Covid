from math import pi

import pandas as pd

from bokeh.io import show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
import datetime
from os.path import dirname, join

import pandas as pd
from scipy.signal import savgol_filter

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, DataRange1d, Select,HoverTool, HBar
from bokeh.palettes import GnBu3, OrRd3
from bokeh.palettes import Blues4
from bokeh.plotting import figure
import numpy as np
from datetime import date
from bokeh.palettes import mpl, d3,brewer
from bokeh.io import output_file, show
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter)
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import transform

interventions = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']


years = list(data.index)
months = list(data.columns)

# reshape to 1D array or rates with a month and year for each row.
df = pd.read_csv("data/data.csv")
df = df[df["CountryName"]=="Italy"]
dates = list(df["Date"].astype(str).unique())
# this is the colormap from the original NYTimes plot
colors = list(brewer['RdYlGn'][4])
colors[1] = '#FDE724'
colors[2] = '#FBA40A'
mapper = LinearColorMapper(palette=colors, low=df["C1_School closing"].min(), high=df["C2_Workplace closing"].max())

p = figure(title="Interventions",
           x_range=dates, y_range=interventions,
           x_axis_location=None, plot_width=1500, plot_height=800,
           tools="", toolbar_location='below')

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "12px"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

#p.rect(x="Date", y=0, width=1, height=1,
#       source=df,line_color = transform("C1_School closing", mapper),fill_color=transform("C1_School closing", mapper))

for i,j in enumerate(interventions):
	p.rect(x="Date", y=i+0.5, width=1, height=1, source=df,
line_color = "black",fill_color=transform(j, mapper))

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="12px",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     label_standoff=6, border_line_color=None, location=(0, 0))
p.add_layout(color_bar, 'right')

show(p) 
