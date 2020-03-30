from ._anvil_designer import Form1Template
from anvil import *
import anvil.server
import anvil.image

class Form1(Form1Template):
  def __init__(self, **properties):
    # Set Form properties and Data Bindings.
    self.init_components(**properties)
    

    # Any code you write here will run when the form opens.

  def file_loader_1_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    self.image_1.source = file
    combined_image = anvil.server.call('findVirus',file)
    self.image_2.source = combined_image
    self.link_2.url = combined_image
    self.card_3.visible = True
    self.card_4.visible = True
    pass
  

  
