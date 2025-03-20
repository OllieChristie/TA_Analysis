# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:52:15 2025

@author: Ollie
"""

import tkinter as tk
from tkinter import ttk, filedialog, Menu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PumpProbeReader import PumpProbeReader
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import RectangleSelector
from scipy.interpolate import interpn
import os

class BackgroundSubtractionApp:
    def __init__(self, root):
        
        #----------------------------------------------------------------------
        # Initialise app-------------------------------------------------------
        #----------------------------------------------------------------------
        
        # Placeholder data
        #----------------------------------------------------------------------

        self.data = None
        self.x1 = None
        self.Wavx1 = None
        self.SelectedWavelength = None
        self.colorbar1 = None
        self.colorbar2 = None
        self.vline = None
        self.time_offset = None
        self.Refdata = None
        self.nContours = 0
        self.SpecWavelength = 532
        self.time_offset = 0
        hovering_over_graph = False
        self.bckgSubtractedSpecMax = None
        self.zoom = 0
        self.retain_colorbar = 0
        self.norm = None
        self.BlankScaling = 1
        
        # Default root properties
        #----------------------------------------------------------------------
        self.root = root
        self.root.title("Contour Plot Viewer")
        self.root.config(bg='white')
        self.root.state('zoomed')
        
        # Configure the grid to expand
        #----------------------------------------------------------------------
        number_of_rows = 10
        number_of_columns = 8
        # Configure rows
        for i in range(number_of_rows):
            self.root.grid_rowconfigure(i, weight=1) # Configures the rate of expansion of each row.
                                                     # Can be set to grow with different scaling factors but want same layout no matter window size.
        
        # Configure columns
        for j in range(number_of_columns):
            self.root.grid_columnconfigure(j, weight=1) # As above for rows
            
        #----------------------------------------------------------------------
        # Menu Bar ------------------------------------------------------------
        #----------------------------------------------------------------------
            
        menu_bar = Menu(root)

        # Create the File menu
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New", command=lambda: print("New File"))
        file_menu.add_command(label="Open", command=lambda: print("Open File"))
        file_menu.add_command(label="Save", command=self.save_matrix_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        # Add the File menu to the menu bar
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Create the Edit menu
        edit_menu = Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Cut", command=lambda: print("Cut"))
        edit_menu.add_command(label="Copy", command=lambda: print("Copy"))
        edit_menu.add_command(label="Paste", command=lambda: print("Paste"))
        
        # Add the Edit menu to the menu bar
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        
        root.config(menu=menu_bar)
        
        #Right-click Menu
        #----------------------------------------------------------------------
        self.right_click_menu = Menu(root, tearoff=0)
        self.right_click_menu.add_command(label='Update Color Scale', command=self.update_colorscale)
        root.bind('<Button-3>', self.on_right_click)
        
        
        #----------------------------------------------------------------------
        # Buttons--------------------------------------------------------------
        #----------------------------------------------------------------------

        self.data_buttons_frame = ttk.Frame(root)
        self.data_buttons_frame.grid(row=1, column=2, padx=1, pady=1)

        #Load Data
        #----------------------------------------------------------------------

        self.load_data_btn = ttk.Button(self.data_buttons_frame, text="Load Data", command=self.load_data)
        self.load_data_btn.pack()
        
        #Load Reference
        #----------------------------------------------------------------------

        self.load_ref_btn = ttk.Button(self.data_buttons_frame, text="Load Reference", command=self.load_reference)
        self.load_ref_btn.pack()
        
        #----------------------------------------------------------------------
        # Spinboxes------------------------------------------------------------
        #----------------------------------------------------------------------
        
        # Spinbox for selecting wavelength slice
        #----------------------------------------------------------------------
        self.wavelength_step_Frame = ttk.Frame(root)
        self.wavelength_step_Frame.grid(row=2, column=2, padx=10, pady=10)
        
        self.wavelength_step_Label = ttk.Label(self.wavelength_step_Frame, text='Wavelength (nm)')
        self.wavelength_step_Label.pack()
        
        self.wavelength_step = tk.DoubleVar()
        
        self.spinbox_min = 400 #Arbitrary starting point. Will default to average wavelength after data loading
        self.spinbox_max = 600
        
        self.wavelength_step_spinbox = ttk.Spinbox(self.wavelength_step_Frame,
                                                   from_=self.spinbox_min, to=self.spinbox_max,
                                                   textvariable=self.wavelength_step,
                                                   command=self.update_slice,
                                                   increment=2) # May cause issues if data wavelength points are more widely spaced than increment
        
        self.wavelength_step_spinbox.bind('<Return>', lambda event: self.update_slice()) #Allows manual input of wavelengths
        self.wavelength_step_spinbox.pack()

        
        
        # Spinbox for selecting timestep (ps)
        #----------------------------------------------------------------------
        self.timeoffset_step_Frame = ttk.Frame(root)
        self.timeoffset_step_Frame.grid(row=2, column=3, padx=10, pady=10)
        
        self.timeoffset_step_Label = ttk.Label(self.timeoffset_step_Frame, text='Time Offset (ps)')
        self.timeoffset_step_Label.pack()
        
        self.timeoffset_step = tk.DoubleVar() 
        self.timeoffset_min = -10
        self.timeoffset_max = 10
        self.timeoffset_step_spinbox = ttk.Spinbox(self.timeoffset_step_Frame,
                                                   from_=self.timeoffset_min, to=self.timeoffset_max,
                                                   textvariable=self.timeoffset_step,
                                                   command=self.update_time_offset,
                                                   increment=0.01) # May cause issues if data wavelength points are more widely spaced than increment
        
        self.timeoffset_step_spinbox.bind('<Return>', lambda event: self.update_time_offset()) #Allows manual input of wavelengths
        self.timeoffset_step_spinbox.pack()
        
        # Spinbox scaling blank
        self.BlankScale_Label = ttk.Label(self.timeoffset_step_Frame, text='Blank Scaling')
        self.BlankScale_Label.pack()
        
        self.BlankScale_step = tk.DoubleVar() 
        self.BlankScale_min = -10
        self.BlankScale_max = 10
        self.BlankScale_spinbox = ttk.Spinbox(self.timeoffset_step_Frame,
                                                   from_=self.BlankScale_min, to=self.BlankScale_max,
                                                   textvariable=self.BlankScale_step,
                                                   command=self.update_BlankScaling,
                                                   increment=0.05) # May cause issues if data wavelength points are more widely spaced than increment
        
        self.BlankScale_spinbox.bind('<Return>', lambda event: self.update_BlankScaling()) #Allows manual input of blank scaling
        self.BlankScale_spinbox.pack()
        
        #----------------------------------------------------------------------
        
        #Contour control
        #----------------------------------------------------------------------
        self.contour_step_Frame = ttk.Frame(root)
        self.contour_step_Frame.grid(row=1, column=3, padx=10, pady=10)
        
        self.contour_step_Label = ttk.Label(self.contour_step_Frame, text = 'No. Contour Lines')
        self.contour_step_Label.pack()
        
        self.contour_step = tk.DoubleVar()
        
        self.contour_min = 0 #Arbitrary starting point. Will default to average wavelength after data loading
        self.contour_max = 20
        
        self.contour_step_spinbox = ttk.Spinbox(self.contour_step_Frame,
                                                   from_=self.contour_min, to=self.contour_max,
                                                   textvariable=self.contour_step,
                                                   command=self.update_bckgSubtraction,
                                                   increment=1) # May cause issues if data wavelength points are more widely spaced than increment
        
        self.contour_step_spinbox.bind('<Return>', lambda event: self.update_bckgSubtraction()) #Allows manual input of wavelengths
        self.contour_step_spinbox.pack()
        
        #----------------------------------------------------------------------
        # Tickboxes------------------------------------------------------------
        #----------------------------------------------------------------------
        
        self.WavSliceBS = tk.BooleanVar()
        self.WavSliceBS.set(False)
               
        self.bckgSub_Slice_Tickbox_Frame = ttk.Frame(self.timeoffset_step_Frame)
        
        self.bckgSub_Slice_Tickbox_Label = ttk.Label(self.bckgSub_Slice_Tickbox_Frame, text='BS?') 
        self.bckgSub_Slice_Tickbox_Label.pack()
        
        self.bckgSub_Slice_Tickbox = ttk.Checkbutton(self.bckgSub_Slice_Tickbox_Frame, variable = self.WavSliceBS, command=self.update_slice)
        self.bckgSub_Slice_Tickbox.pack()
        
        self.bckgSub_Slice_Tickbox_Frame.pack()
        
        #----------------------------------------------------------------------
        #-Slider Bars----------------------------------------------------------
        #----------------------------------------------------------------------
                
        #Wavelength Slider
        #----------------------------------------------------------------------
        
        self.wavelength_slider = tk.Scale(self.wavelength_step_Frame, from_=self.spinbox_min, to=self.spinbox_max, orient=tk.HORIZONTAL, variable=self.SpecWavelength, command=self.update_spinbox, length = 400)
        self.wavelength_slider.pack()
        
        #----------------------------------------------------------------------
        # Plots----------------------------------------------------------------
        #----------------------------------------------------------------------
        
        # Raw 2D data
        #----------------------------------------------------------------------

        self.fig1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=root)
        self.canvas1.get_tk_widget().grid(row=0, column=6, padx=10, pady=20, rowspan=4, columnspan=4)
        self.ax1.set_title('Raw Data')
        self.ax1.set_xlabel('Wavelength (nm)')
        self.ax1.set_ylabel('Time (ps)')
        self.ax1.set_yscale('symlog')
        plt.tight_layout()
                
        # 2D Background Subtraction
        #----------------------------------------------------------------------

        self.fig2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=root)
        self.canvas2.get_tk_widget().grid(row=5, column=6, padx=10, pady=10, rowspan=4, columnspan=4)
        self.canvas2.get_tk_widget().bind("<Enter>", self.on_enter)
        self.canvas2.get_tk_widget().bind("<Leave>", self.on_leave)
        self.ax2.set_title('Background Subtraction')
        self.ax2.set_xlabel('Wavelength (nm)')
        self.ax2.set_ylabel('Time (ps)')
        self.ax2.set_yscale('symlog')
        plt.tight_layout()
        
        
        # Wavelength slice figure
        #----------------------------------------------------------------------

        self.fig3, self.ax3 = plt.subplots()
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=root)
        self.canvas3.get_tk_widget().grid(row=5, column=1, padx=10, pady=10, rowspan=4, columnspan=4)
        self
        self.ax3.set_xlabel('Time (ps)')
        self.ax3.set_ylabel('PIA (a.u.)')
        self.ax3.set_xscale('symlog')
        plt.tight_layout()
        
        #----------------------------------------------------------------------
        # Zooming--------------------------------------------------------------
        #----------------------------------------------------------------------
        
        # Add a RectangleSelector for zoom functionality
        self.rect_selector1 = RectangleSelector(self.ax1, self.on_select_zoom, useblit=True)
        self.rect_selector2 = RectangleSelector(self.ax2, self.on_select_zoom, useblit=True)
        self.rect_selector3 = RectangleSelector(self.ax3, self.on_select_zoom_WavSlice, useblit=True)
        
        # Bind double-click event to reset zoom
        self.canvas1.mpl_connect('button_press_event', self.on_double_click)
        self.canvas2.mpl_connect('button_press_event', self.on_double_click)
        self.canvas3.mpl_connect('button_press_event', self.on_double_click_WavSlice)
        
    #--------------------------------------------------------------------------
    #Functions-----------------------------------------------------------------
    #--------------------------------------------------------------------------


    def load_data(self):
        """Load a 2D numerical dataset."""
        
        #Clear data if pre-existing file has been loaded
        #----------------------------------------------------------------------
        if self.data is not None: 
            self.data = None
            self.time = None
            self.Wavelength = None
            self.current_datafile = None
            
        #Open file explorer and load data
        #----------------------------------------------------------------------
        file_path = filedialog.askopenfilename()
        file_name = os.path.splitext(file_path)[0]
        if file_path:
            dataRead = PumpProbeReader() #Extract data from matrix file
            dataRead.OpenMatrixFile(file_path)
            
            self.time = dataRead.ProbeTime # self.time will be variable and used ot plot spectra after changes
            self.ProbeTime0 = dataRead.ProbeTime    # self.ProbeTime0 will be unchanged and refer to raw time points

            self.Wavelength = dataRead.Wavelength # self.Wavelength shouldn't be changed in current version. 
                                                  # May need integrating if wavelength shifts implemented
                                                  
            self.data = dataRead.Spectra  # 2D dataset
            self.data0 = self.data #Stores original data
            
            self.current_datafile = file_name
            
            self.zoomed_data = self.data
            self.zoomlowerXInd = 0
            self.zoomlowerYInd = 0
            self.zoomupperXInd = np.shape(self.data)[1]
            self.zoomupperYInd = np.shape(self.data)[0]

            #Change wavelength spinbox to accommodate raw data wavelengths
            #------------------------------------------------------------------
            self.spinbox_min = np.min(self.Wavelength)
            self.spinbox_max = np.max(self.Wavelength)
            
            self.wavelength_step_spinbox.config(from_=self.spinbox_min, to=self.spinbox_max)
            self.wavelength_slider.config(from_=self.spinbox_min, to=self.spinbox_max)
            
            #Check for loaded reference data and plot 2D datasets.
            if self.Refdata is not None:
                self.plot_contour() # Will plot 2D reference data if no raw data is loaded.
                self.plot_difference_contour() # Will only plot if both datasets loaded.
            else:            
                self.plot_contour() # Will plot raw 2D data but won't attempt background subtraction without reference.
            
    
    def load_reference(self):
        """Load a reference dataset."""
        file_path = filedialog.askopenfilename()
        if file_path:
            RefRead = PumpProbeReader()
            RefRead.OpenMatrixFile(file_path)
            self.Reftime = RefRead.ProbeTime
            self.RefWavelength = RefRead.Wavelength
            RefSpectra = RefRead.Spectra
            
            self.spinbox_min = np.min(self.RefWavelength)
            self.spinbox_max = np.max(self.RefWavelength)
            
            self.wavelength_step_spinbox.config(from_=self.spinbox_min, to=self.spinbox_max)
            
            self.Refdata = RefSpectra  # Placeholder for actual data loading
            self.Refdata0 = RefSpectra
            self.zoomed_ref = self.Refdata # Initialise for zooming

            
            if self.data is None:
                self.data = self.Refdata
                # self.time = self.Reftime
                self.Wavelength = self.RefWavelength
                
                self.plot_contour()
            else:
                self.plot_difference_contour()
                
            self.update_slice()                

    def plot_contour(self):
        """Plot the contour of the dataset."""
        if self.data is not None:
            
            # Store the original limits for resetting zoom
            #------------------------------------------------------------------
            
            if self.colorbar1:
                self.colorbar1.remove()
                
            self.ax1.clear()

            self.ax1.set_title('Raw Data')
            self.ax1.set_xlabel('Wavelength (nm)')
            self.ax1.set_ylabel('Time (ps)')
            self.ax1.set_yscale('symlog')
            
            Spectra = self.data
            
            normalisedSpectrum = Spectra/np.max(Spectra)
            minSpec = np.min(normalisedSpectrum)
            
            zeroAlignedSpectrum = normalisedSpectrum+abs(minSpec)
            zeroPoint = abs(minSpec/np.max(zeroAlignedSpectrum))
            
            gradient = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'violet'), (zeroPoint-(zeroPoint/2), 'blue'), (zeroPoint, 'Black'), (zeroPoint+((1-zeroPoint)/2), 'red'), (1, 'yellow')])
            
            X, Y = np.meshgrid(self.Wavelength, self.time)
            self.cp1 = self.ax1.contourf(X, Y, self.data, 500, cmap = gradient)
            self.ax1.contour(X, Y, self.data, self.nContours, colors='black', linewidths=1)
            self.colorbar1 = self.fig1.colorbar(self.cp1, ax=self.ax1)
            self.ax1.set_xlim(np.min(self.Wavelength), np.max(self.Wavelength))
            self.original_xlim, self.original_ylim = self.ax1.get_xlim(), self.ax1.get_ylim()
            if self.x1 is not None:
                self.ax1.set_xlim(min(self.x1, self.x2), max(self.x1, self.x2))
                self.ax1.set_ylim(min(self.y1, self.y2), max(self.y1, self.y2))


            plt.tight_layout()

            self.canvas1.draw()
            plt.tight_layout()
            
            self.update_slice()
            
    def plot_difference_contour(self):
        #Background subtraction
        # if self.time_offset is not None:
        #     self.time = self.ProbeTime0-self.time_offset
        
        #If new loaded data is different size to currently loaded reference, 
        #catch Type Error and clear reference
        #----------------------------------------------------------------------
        try:
            bckgSubtractedSpec = self.data-self.BlankScaling*self.Refdata
        except TypeError:
            self.ax2.clear()
            self.Refdata = None
            self.Reftime = None
            self.RefWavelength = None
            pass
        
        #Plotting        
        if self.colorbar2:
            self.colorbar2.remove()
            
        self.ax2.clear()
        
        self.ax2.set_title('Background Subtraction')
        self.ax2.set_xlabel('Wavelength (nm)')
        self.ax2.set_ylabel('Time (ps)')
        self.ax2.set_yscale('symlog')
        
        self.BackgroundSubtraction = bckgSubtractedSpec
        
        X, Y = np.meshgrid(self.Wavelength, self.time)
        Spectra = self.BackgroundSubtraction
        
        if self.retain_colorbar == 0:
            self.find_zeropoint()
            gradient = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'violet'), (self.zeroPoint-(self.zeroPoint/2), 'blue'), (self.zeroPoint, 'Black'), (self.zeroPoint+((1-self.zeroPoint)/2), 'red'), (1, 'yellow')])
        else:
            gradient = self.gradient
        
        if self.norm is None:
            self.cp2 = self.ax2.contourf(X, Y, Spectra, 500, cmap = gradient)
            self.ax2.contour(X, Y, Spectra, self.nContours, colors='black', linewidths=1)
            self.colorbar2 = self.fig2.colorbar(self.cp2, ax=self.ax2)
        else:
            self.cp2 = self.ax2.contourf(X, Y, Spectra, 500, cmap = gradient, norm = self.norm)
            self.ax2.contour(X, Y, Spectra, self.nContours, colors='black', linewidths=1)
            self.colorbar2 = self.fig2.colorbar(self.cp2, ax=self.ax2)
        
        if self.retain_colorbar == 0:
            self.update_colorscale()
        
        if self.x1 is not None:
            self.ax2.set_xlim(min(self.x1, self.x2), max(self.x1, self.x2))
            self.ax2.set_ylim(min(self.y1, self.y2), max(self.y1, self.y2))
        else:
            self.ax2.set_xlim(np.min(self.Wavelength), np.max(self.Wavelength))
        
        plt.tight_layout()

        self.canvas2.draw()
        plt.tight_layout()    
            
    def plot_wav_line(self):
        if self.SelectedWavelength is not None:
            if self.vline:
                self.vline.remove()
            self.vline = self.ax1.axvline(self.SelectedWavelength, linewidth = 2, color='white', linestyle='--')
            self.canvas1.draw_idle()
    
    def update_slice(self):
        """Update the wavelength slice dynamically."""
        if self.data is not None:
            self.SpecWavelength = self.wavelength_step.get()
            
            if min(self.Wavelength) <= self.SpecWavelength <= max(self.Wavelength):
               # dummy = 0 
               pass
            else:
                self.SpecWavelength = np.mean(self.Wavelength)                    
                
            
            WavIndex = np.argmin(abs(self.Wavelength-self.SpecWavelength))
            self.SelectedWavelength = self.Wavelength[WavIndex]
                        
            self.wavelength_step_spinbox.delete(0, tk.END)
            self.wavelength_step_spinbox.insert(0, f"{self.SelectedWavelength:.4g}")
            if 0 <= WavIndex < self.data.shape[1]:
                self.ax3.clear()
                self.ax3.set_xlabel('Time (ps)')
                self.ax3.set_ylabel('PIA (a.u.)')
                self.ax3.set_xscale('symlog')
                if self.Refdata is not None:
                    self.ax3.plot(self.Reftime, self.Refdata[:, WavIndex], c='red', linestyle = '--', linewidth=1)
                    self.ax3.plot(self.Reftime, self.data[:, WavIndex], c='red', linewidth = 1.5) # Need to plot on same time axis as reference for time correction.
                    if self.WavSliceBS.get():
                        self.ax3.plot(self.time, self.BackgroundSubtraction[:, WavIndex], c='black', linewidth = 1.5) # Plotting if reference not loaded in

                else:
                    self.ax3.plot(self.time, self.data[:, WavIndex], c='red', linewidth = 1.5) # Plotting if reference not loaded in


                self.ax3.set_xlim(np.min(self.time), np.max(self.time))
                self.ax3.set_xlim(np.min(self.ProbeTime0), np.max(self.ProbeTime0))
                self.original_Wavxlim, self.original_Wavylim = self.ax3.get_xlim(), self.ax3.get_ylim()
                
                if self.Wavx1 is not None:
                    self.ax3.set_xlim(min(self.Wavx1, self.Wavx2), max(self.Wavx1, self.Wavx2))
                    self.ax3.set_ylim(min(self.Wavy1, self.Wavy2), max(self.Wavy1, self.Wavy2))
                    
                self.plot_wav_line()
                self.ax3.set_title(f"Wavelength Slice at {self.SelectedWavelength:.4g} nm")
                plt.tight_layout()

                self.canvas3.draw()
                plt.tight_layout()
                
    def update_bckgSubtraction(self):
        self.nContours = int(self.contour_step.get())
        self.plot_contour()
        if self.Refdata is not None:
            self.plot_difference_contour()
            
            # Add a method to handle zoom on selection
    def on_select_zoom(self, eclick, erelease):
        if eclick.button == 1:
            self.x1, self.y1 = eclick.xdata, eclick.ydata
            self.x2, self.y2 = erelease.xdata, erelease.ydata
            
            xlim = min(self.x1, self.x2), max(self.x1, self.x2) 
            ylim = min(self.y1, self.y2), max(self.y1, self.y2)
            
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim)
            self.ax2.set_xlim(xlim)
            self.ax2.set_ylim(ylim)
            self.canvas1.draw()
            self.canvas2.draw()
            
            self.zoomlowerXInd = np.argmin(abs(self.Wavelength-xlim[0]))
            self.zoomupperXInd = np.argmin(abs(self.Wavelength-xlim[1]))
            
            self.zoomlowerYInd = np.argmin(abs(self.time-ylim[0]))
            self.zoomupperYInd = np.argmin(abs(self.time-ylim[1]))
            self.zoom = 1
            
    def on_select_zoom_WavSlice(self, eclick, erelease):
        self.Wavx1, self.Wavy1 = eclick.xdata, eclick.ydata
        self.Wavx2, self.Wavy2 = erelease.xdata, erelease.ydata
        self.ax3.set_xlim(min(self.Wavx1, self.Wavx2), max(self.Wavx1, self.Wavx2))
        self.ax3.set_ylim(min(self.Wavy1, self.Wavy2), max(self.Wavy1, self.Wavy2))
        self.canvas3.draw()

   
        # Add methods to reset zoom on double-click
    def on_double_click(self, event):
        if event.dblclick:
            
            self.ax1.set_xlim(self.original_xlim)
            self.ax1.set_ylim(self.original_ylim)
            self.ax2.set_xlim(self.original_xlim)
            self.ax2.set_ylim(self.original_ylim)
            self.canvas1.draw()
            self.canvas2.draw()   
            
            self.x1 = None
            self.x2 = None
            self.y1 = None
            self.y2 = None
            
            self.zoomlowerXInd = None
            self.zoomlowerYInd = None
            self.zoomupperXInd = None
            self.zoomupperYInd = None
            self.zoom = 0
                        
    def on_double_click_WavSlice(self, event):
        if event.dblclick:
            self.ax3.set_xlim(self.original_Wavxlim)
            self.ax3.set_ylim(self.original_Wavylim)

            self.canvas3.draw()
  
            self.Wavx1 = None
            self.Wavx2 = None
            self.Wavy1 = None
            self.Wavy2 = None
                    

    def on_right_click(self, event):
        if hovering_over_graph:
            self.right_click_menu.entryconfig("Update Color Scale", state="normal")
        else:
            self.right_click_menu.entryconfig("Update Color Scale", state="disabled")
        self.right_click_menu.post(event.x_root, event.y_root)
    
    def on_enter(self,event):
        global hovering_over_graph
        hovering_over_graph = True
    
    def on_leave(self,event):
        global hovering_over_graph
        hovering_over_graph = False 
        
    def update_spinbox(self, value):
        self.SpecWavelength = self.wavelength_slider.get()
        self.wavelength_step_spinbox.set(self.SpecWavelength)
        self.update_slice()
        
    def update_time_offset(self):
        self.time_offset = self.timeoffset_step.get()
        self.time = self.ProbeTime0 - self.time_offset
        self.time[0] = self.ProbeTime0[0]
        self.time[-1] = self.ProbeTime0[-1]
        
        ProbeTime0 = np.array(self.ProbeTime0).reshape(-1,1)
        
        points = (ProbeTime0.flatten(),)
        interpdata = np.zeros((len(self.time), self.data.shape[1]))
        
        for i in range(self.data.shape[1]):
            interpdata[:,i] = interpn(points, self.data0[:,i], self.time)
            
        self.data = interpdata
        
        self.update_slice()
        self.retain_colorbar = 1 
        self.plot_difference_contour()
        self.retain_colorbar = 0
        
    def update_BlankScaling(self):
        self.BlankScaling = self.BlankScale_step.get()
        self.Refdata = self.BlankScaling*self.Refdata0
        self.plot_difference_contour()
        self.update_slice()
        
        
    def update_colorscale(self):
        Spectrum0 = self.BackgroundSubtraction[self.zoomlowerYInd:self.zoomupperYInd, self.zoomlowerXInd:self.zoomupperXInd]
        vmin = np.min(Spectrum0)
        vmax = np.max(Spectrum0)
        self.norm = Normalize(vmin=vmin, vmax=vmax)
        self.cp2.set_norm(self.norm)
        self.find_zeropoint()
        self.gradient = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'violet'), (self.zeroPoint-(self.zeroPoint/2), 'blue'), (self.zeroPoint, 'Black'), (self.zeroPoint+((1-self.zeroPoint)/2), 'red'), (1, 'yellow')])
        self.gradient.set_over('none')  # Set color for values above vmax
        self.gradient.set_under('none')  
        self.cp2.set_cmap(self.gradient)
        self.colorbar2.update_normal(self.cp2)
        self.canvas2.draw()
        
    def save_matrix_file(self):
        
        FileSave = PumpProbeReader()
        FileSave.ProbeTime = self.Reftime
        FileSave.Spectra = self.BackgroundSubtraction
        FileSave.Wavelength = self.Wavelength
        save_file_name = self.current_datafile+'_BS.dat'
        FileSave.WriteMatrixFile(save_file_name)
        
    def find_zeropoint(self):        
        if self.zoom == 0:
            
            Spectrum0 = self.BackgroundSubtraction
            
        else:
            
            Spectrum0 = self.BackgroundSubtraction[self.zoomlowerYInd:self.zoomupperYInd, self.zoomlowerXInd:self.zoomupperXInd]
            
        normalisedSpectrum = Spectrum0/np.max(Spectrum0)
        minSpec = np.min(normalisedSpectrum)
        
        zeroAlignedSpectrum = normalisedSpectrum-minSpec
        zeroPoint = abs(minSpec/np.max(zeroAlignedSpectrum))
        self.zeroPoint = zeroPoint
        
    def RunCode(self):
        self.root.mainloop()
        
# class RunCode:     
#     # if __name__ == "__main__":
#         root = tk.Tk()
#         app = BackgroundSubtractionApp(root)
#         root.mainloop()