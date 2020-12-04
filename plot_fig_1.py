from bioviz import Viz

b = Viz(model_path="arm_wt_rot_scap.bioMod")
b.vtk_window.change_background_color((1, 1, 1))
b.exec()
