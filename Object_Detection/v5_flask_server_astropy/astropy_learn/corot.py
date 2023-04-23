from bokeh.palettes import Magma
from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook, push_notebook

p = figure(
    title="Some title",
    plot_width=400,
    plot_height=600)

# Style the figure image
p.grid.grid_line_alpha = 0.1
p.xgrid.band_fill_alpha = 0.1
p.xgrid.band_fill_color = Magma[10][1]
p.yaxis.axis_label = "Some label for y axis"
p.xaxis.axis_label = "Some label for x axis"
hdulist = pyfits.open('./sat_00000.0101.fits')
# hdulist = pyfits.open('./gimg-0900.fits')
infos = hdulist.info()
img_data = hdulist[0].data
# Place the information on plot
p.line(x_data, y_data,
        legend_label="My legend label",
        line_width=2,
        color=Magma[10][2],
        muted_alpha=0.1,
        line_cap='rounded')
p.legend.location = "right_top"
p.legend.click_policy = "disable"

show(p)