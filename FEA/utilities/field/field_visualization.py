"""
Class FieldMap -- Visual representation of the prescription map.
Includes TkInter functionality and Foium .html representation creation
"""

from field.field_creation import *
from utilities.filereaders import *

import folium
import operator, webbrowser, os


# from tkinter import *
# from tkinter import filedialog


class FieldMap:
    """
    Class that holds the prescription cell list to be visualized.
    Inner objects:
        self.field -- field shape information
        self.cell_list -- List of GridCell objects containing geometry, yield, pro, and nitrogen information.
    """

    def __init__(self, field, prescription=None):
        """
        Parameters:
            field -- field_representation.field class: unoptimized prescription in 'cell_list' attribute
            prescription -- GA_code.Prescription class: optimized prescription in 'cell_list' attribute
        """
        self.field = field
        if prescription is not None:
            self.cell_list = prescription.cell_list
        else:
            self.cell_list = field.cell_list
        self.adjust_cell_window = None  # TkInter UI cell adjustment option

    def create_folium_map(self):
        """
        Uses 'folium' package to create .html template to create map visualization
        """
        reference_cell_top = self.cell_list[0]
        reference_cell_bottom = self.cell_list[-1]
        field_map = folium.Map([(reference_cell_top.upperright_y + reference_cell_bottom.bottomleft_y) / 2,
                                (reference_cell_bottom.bottomleft_x + reference_cell_top.upperright_x) / 2],
                               zoom_start=16, width='65%', height='65%')

        # applies field shape and field grid to the map
        folium.Choropleth(geo_data=self.field.field_shape, data=None, columns=['geometry'], line_opacity=1,
                          fill_opacity=0.4, fill_color="Grey").add_to(field_map)

        N_colors = ["#a10000", "#a15000", "#a1a100", "#416600", "#008282", "#005682"]

        self.cell_list.sort(key=operator.attrgetter('sorted_index'))
        for cell in self.cell_list:
            self.field.nitrogen_list.sort()
            fill_color = str(N_colors[self.field.nitrogen_list.index(cell.nitrogen)])
            # Create folium 'Rectangles' to display on the map.
            # Pop-up string determines what information to show: only shows protein and yield if calculated for cell.
            # TODO: adjust to take any shape, not just rectangles
            if cell.pro_ != -1 and cell.yield_ != -1:
                popup_string = '<div class="nitrogen-cell value-%d" id="%d">Previous year protein: %.2f <br> Previous year yield: %.2f <br> Fertilizer prescribed: %d <br> Plot index: %d</div>' % (
                    cell.nitrogen, cell.sorted_index,cell.pro_, cell.yield_, cell.nitrogen, cell.sorted_index)
            elif cell.yield_ != -1:
                popup_string = '<div class="nitrogen-cell value-%d" id="%d">Previous year yield: %.2f <br> Fertilizer prescribed: %d <br> Plot index: %d</div>' % (
                    cell.nitrogen, cell.sorted_index,cell.yield_, cell.nitrogen, cell.sorted_index)
            else:
                popup_string = '<div class="nitrogen-cell value-%d" id="%d">Fertilizer prescribed: %d <br> Plot index: %d</div>' % (cell.nitrogen, cell.sorted_index, cell.nitrogen, cell.sorted_index)
            folium.vector_layers.Rectangle(bounds=cell.folium_bounds, popup=folium.Popup(html=popup_string, parse_html=False), color="black", weight=0.1,
                                           fill_color=fill_color, fill_opacity=.8).add_to(field_map)

        # Create nitrogen legend pop-up marker
        html_string = """Fertilizer Legend <br>"""
        i = 0
        while i < len(self.field.nitrogen_list):
            opening = "<font size=1 color = " + N_colors[i] + ">"
            new_string = "&#9608 Fertilizer level of %d" % (self.field.nitrogen_list[i])
            full_string = opening + new_string + "</font>" + "<br>"
            html_string = html_string + full_string
            i += 1
        folium.map.Marker(
            # places marker 1000 feet above and 100 feet to the right of the bottom right corner of the field
            [self.field.field_shape.bounds[1] + (1000 / 364567.2), self.field.field_shape.bounds[2] + (100 / 364567.2)],
            icon=folium.features.DivIcon(
                icon_size=(150, 50),
                icon_anchor=(0, 0),
                html=html_string,
            )
        ).add_to(field_map)
        folium.LatLngPopup().add_to(field_map)

        parent_dir = os.path.realpath('.')  # determine application home directory
        path_ = str(os.path.abspath('./app/templates/folium_prescription.html'))
        field_map.save(path_)  # Saves .html page to templates dir str(parent_dir) + '//app//templates//prescription_map_2.html'
        # TkInter UI option to save and open .html file
        #   field_map.save("Field_" + str(self.field.id)+"_best_map.html")
        #   webbrowser.open_new_tab("Field_" + str(self.field.id)+"_best_map.html")

    # def start_adjust_cells_window(self):
    #     self.adjust_cell_window = Tk()
    #     self.create_adjust_cells_window()
    #     self.adjust_cell_window.mainloop()

    # def create_adjust_cells_window(self):
    #     self.adjust_cell_window.lift()
    #     self.adjust_cell_window.title("Change Cell Rates")

    #     change_frame = Frame(self.adjust_cell_window)
    #     change_frame.grid(row=0, columnspan=4)
    #     finish_frame = Frame(self.adjust_cell_window)
    #     finish_frame.grid(row=2, columnspan=4)

    #     field_label = Label(change_frame, text="field " + str(self.field.id))
    #     field_label.grid(row=0, columnspan=2)

    #     entry_label = Label(change_frame, text="Entry cell: ")
    #     cell_index = Entry(change_frame)
    #     entry_label.grid(row=1, column=0)
    #     cell_index.grid(row=1, column=1)

    #     nitrogen_label = Label(change_frame, text="Nitrogen value: ")
    #     nitrogen_value = Entry(change_frame)
    #     nitrogen_label.grid(row=2, column=0)
    #     nitrogen_value.grid(row=2, column=1)

    #     runSubmit = Button(change_frame, text="Save change", command=lambda:
    #                   self.change_cell_nitrogen(int(cell_index.get()), int(nitrogen_value.get())))
    #     runSubmit.grid(row=3, columnspan=2)

    #     mapButton = Button(change_frame, text="Generate new map",
    #                        command=lambda: self.update_map())
    #     mapButton.grid(row=4, columnspan=2)

    #     download_button = Button(finish_frame, text="Download", command=lambda: self.download_map(
    #         filedialog.asksaveasfilename(title="Select file", filetypes=(("csv files", "*.csv"),
    #         ("all files", "*.*")))))
    #     download_button.grid(row=0, column=0)
    #     satisfiedButton = Button(finish_frame, text="Quit", command=lambda: self.close_window())
    #     satisfiedButton.grid(row=0, column=1)

    def change_cell_nitrogen(self, cell_index, nitrogen_value):
        """
        Change the current prescription for a specific cell.
        Takes the cell index of the cell to be adjusted and the new prescription value.
        """
        for cell in self.cell_list:
            if cell.sorted_index == cell_index:
                cell.nitrogen = nitrogen_value
                break

    def download_map(self, filename):
        """
        Transforms current prescription to CSV file with WKT column and prescription information.
        filename -- name the file should be saved as
        """
        WKTFiles.create_wkt_file(filename, self.cell_list, self.field.field_shape)

    """Functions for TkInter UI window updates """
    def update_map(self):
        self.create_folium_map()
        self.adjust_cell_window.lift()

    def close_window(self):
        self.adjust_cell_window.destroy()
