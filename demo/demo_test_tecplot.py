import os, sys
import numpy as np
import torch
import tecplot

sys.path.append("../")

data_set = "wi3rooms_0"
demo_path = os.path.abspath(os.curdir)
root_path = os.path.abspath(os.path.join(demo_path, ".."))
data_path = os.path.join(root_path, "data", data_set)


def main():
    dataset = tecplot.data.load_tecplot()

    frame = tecplot.active_frame()
    frame.plot_type = tecplot.constant.PlotType.Cartesian3D

    plot = frame.plot()
    plot.vector.u_variable = dataset.variable("U(M/S)")
    plot.vector.v_variable = dataset.variable("V(M/S)")
    plot.contour(0).variable = dataset.variable("T(K)")
    plot.show_streamtraces = True
    plot.show_contour = True
    plot.fieldmap(0).contour.show = True

    # Add streamtraces and set streamtrace style
    streamtraces = plot.streamtraces
    streamtraces.add_rake(
        start_position=(-0.003, 0.005),
        end_position=(-0.003, -0.005),
        stream_type=Streamtrace.TwoDLine,
        num_seed_points=10,
    )

    streamtraces.show_arrows = False
    streamtraces.line_thickness = 0.4

    plot.axes.y_axis.min = -0.02
    plot.axes.y_axis.max = 0.02
    plot.axes.x_axis.min = -0.008
    plot.axes.x_axis.max = 0.04

    tecplot.export.save_png("streamtrace_2D.png", 600, supersample=3)


if __name__ == "__main__":
    main()
