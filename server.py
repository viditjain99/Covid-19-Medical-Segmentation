import anvil.server
import anvil.mpl_util
from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

class SegNetWithSkipConnections(nn.Module):
	def __init__(self):
		super(SegNetWithSkipConnections,self).__init__()
		self.conv1_1=nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
		self.conv1_2=nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)

		self.conv2_1=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
		self.conv2_2=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)

		self.conv3_1=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
		self.conv3_2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
		self.conv3_3=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)

		self.conv4_1=nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1)
		self.conv4_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
		self.conv4_3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)

		self.conv5_1=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
		self.conv5_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
		self.conv5_3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)

		self.upconv5_3=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
		self.upconv5_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
		self.upconv5_1=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)

		self.upconv4_3=nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1)
		self.upconv4_2=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
		self.upconv4_1=nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1)

		self.upconv3_3=nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1)
		self.upconv3_2=nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
		self.upconv3_1=nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)

		self.upconv2_2=nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1)
		self.upconv2_1=nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)

		self.upconv1_2=nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1)
		self.upconv1_1=nn.Conv2d(64,4,kernel_size=3,stride=1,padding=1)

		self.pool1=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
		self.pool2=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
		self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
		self.pool4=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
		self.pool5=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

		self.upsamp5=nn.MaxUnpool2d(kernel_size=2,stride=2)
		self.upsamp4=nn.MaxUnpool2d(kernel_size=2,stride=2)
		self.upsamp3=nn.MaxUnpool2d(kernel_size=2,stride=2)
		self.upsamp2=nn.MaxUnpool2d(kernel_size=2,stride=2)
		self.upsamp1=nn.MaxUnpool2d(kernel_size=2,stride=2)

		self.batch_norm_1=nn.BatchNorm2d(64)
		self.batch_norm_2=nn.BatchNorm2d(128)
		self.batch_norm_3=nn.BatchNorm2d(256)
		self.batch_norm_4=nn.BatchNorm2d(512)

	def forward(self,x):
		#1,1
		dim1=x.size()
		x1=self.conv1_1(x)
		x1=self.batch_norm_1(x1)
		x1=F.relu(x1)
		x1=self.conv1_2(x1)
		x1=self.batch_norm_1(x1)
		x1=F.relu(x1)
		p1,idx1=self.pool1(x1)
		#64,1/2

		#64,1/2
		dim2=p1.size()
		x2=self.conv2_1(p1)
		x2=self.batch_norm_2(x2)
		x2=F.relu(x2)
		x2=self.conv2_2(x2)
		x2=self.batch_norm_2(x2)
		x2=F.relu(x2)
		p2,idx2=self.pool2(x2)
		#128,1/4

		#128,1/4
		dim3=p2.size()
		x3=self.conv3_1(p2)
		x3=self.batch_norm_3(x3)
		x3=F.relu(x3)
		x3=self.conv3_2(x3)
		x3=self.batch_norm_3(x3)
		x3=F.relu(x3)
		x3=self.conv3_3(x3)
		x3=self.batch_norm_3(x3)
		x3=F.relu(x3)
		p3,idx3=self.pool3(x3)
		#256,1/8

		#256,1/8
		dim4=p3.size()
		x4=self.conv4_1(p3)
		x4=self.batch_norm_4(x4)
		x4=F.relu(x4)
		x4=self.conv4_2(x4)
		x4=self.batch_norm_4(x4)
		x4=F.relu(x4)
		x4=self.conv4_3(x4)
		x4=self.batch_norm_4(x4)
		x4=F.relu(x4)
		p4,idx4=self.pool4(x4)
		#512,1/16

		#512,1/16
		dim5=p4.size()
		x5=self.conv5_1(p4)
		x5=self.batch_norm_4(x5)
		x5=F.relu(x5)
		x5=self.conv5_2(x5)
		x5=self.batch_norm_4(x5)
		x5=F.relu(x5)
		x5=self.conv5_3(x5)
		x5=self.batch_norm_4(x5)
		x5=F.relu(x5)
		p5,idx5=self.pool5(x5)
		#512,1/32   

		#512,1/32
		u5=self.upsamp5(p5,idx5,output_size=dim5)
		u5=self.upconv5_3(u5)
		u5=self.batch_norm_4(u5)
		u5=F.relu(u5)
		u5=self.upconv5_2(u5)
		u5=self.batch_norm_4(u5)
		u5=F.relu(u5)
		u5=self.upconv5_1(u5)
		u5=self.batch_norm_4(u5)
		u5=F.relu(u5)
		#512,1/16

		#512,1/16
		u4=self.upsamp4(u5,idx4,output_size=dim4)
		u4=torch.cat([u4,x4],dim=1)
		u4=self.upconv4_3(u4)
		u4=self.batch_norm_4(u4)
		u4=F.relu(u4)
		u4=self.upconv4_2(u4)
		u4=self.batch_norm_4(u4)
		u4=F.relu(u4)
		u4=self.upconv4_1(u4)
		u4=self.batch_norm_3(u4)
		u4=F.relu(u4)
		#256,1/8

		#256,1/8
		u3=self.upsamp3(u4,idx3,output_size=dim3)
		u3=torch.cat([u3,x3],dim=1)
		u3=self.upconv3_3(u3)
		u3=self.batch_norm_3(u3)
		u3=F.relu(u3)
		u3=self.upconv3_2(u3)
		u3=self.batch_norm_3(u3)
		u3=F.relu(u3)
		u3=self.upconv3_1(u3)
		u3=self.batch_norm_2(u3)
		u3=F.relu(u3)
		#128,1/4

		#128,1/4
		u2=self.upsamp2(u3,idx2,output_size=dim2)
		u2=torch.cat([u2,x2],dim=1)
		u2=self.upconv2_2(u2)
		u2=self.batch_norm_2(u2)
		u2=F.relu(u2)
		u2=self.upconv2_1(u2)
		u2=self.batch_norm_1(u2)
		u2=F.relu(u2)
		#64,1/2

		#64,1/2
		u1=self.upsamp1(u2,idx1,output_size=dim1)
		u1=torch.cat([u1,x1],dim=1)
		u1=self.upconv1_2(u1)
		u1=self.batch_norm_1(u1)
		u1=F.relu(u1)
		u1=self.upconv1_1(u1)
		u1=F.softmax(u1,dim=1)
		#4,1

		return u1

def masks_to_labels(masks):
	h=masks.shape[0]
	w=masks.shape[1]
	labels=np.zeros([h,w],dtype='int32')
	for c in range(4):
		for i in range(h):
			for j in range(w):
				if masks[i,j,c]==1:
					labels[i,j]=c
	return labels

colors=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
cm=LinearSegmentedColormap.from_list('rgb_cmap',colors,N=4)
colors_1=[(0,0,0),(0,1,0)]
cm_yellow=LinearSegmentedColormap.from_list('y_cmap',colors_1,N=2)

anvil.server.connect("xxxxxxxxx")

@anvil.server.callable
def findVirus(file):
	img=Image.open(io.BytesIO(file.get_bytes()))
	img=np.asarray(img)
	if len(img.shape)==3:
		img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	img=img/255.0
	img=torch.tensor(img)
	img=img.type(torch.FloatTensor)
	img=img.unsqueeze(0)
	img=img.unsqueeze(0)

	model=SegNetWithSkipConnections()
	model.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu')))

	pred=model(img)
	pred=pred.squeeze(0)
	pred=pred.cpu().detach().numpy()
	pred=np.moveaxis(pred,0,-1)
	pred=pred.round().astype(int)

	imgVirus=masks_to_labels(pred)
	img=img.squeeze(0)
	img=img.squeeze(0)
	img=img.cpu().detach().numpy()

	fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5),sharey=True)
	ax1.axis('off')
	ax1.get_xaxis().set_visible(False)
	ax1.get_yaxis().set_visible(False)
	ax1.set_title('Ground-Glass',fontsize=20)
	ax1.imshow(img,cmap='gray')
	ax1.imshow(pred[:,:,1],cmap=cm_yellow,alpha=0.5)

	ax2.axis('off')
	ax2.get_xaxis().set_visible(False)
	ax2.get_yaxis().set_visible(False)
	ax2.set_title('Consolidation',fontsize=20)
	ax2.imshow(img,cmap='gray')
	ax2.imshow(pred[:,:,2],cmap=cm_yellow,alpha=0.5)

	ax3.axis('off')
	ax3.get_xaxis().set_visible(False)
	ax3.get_yaxis().set_visible(False)
	ax3.set_title('Pleural Effusion',fontsize=20)
	ax3.imshow(img,cmap='gray')
	ax3.imshow(pred[:,:,3],cmap=cm_yellow,alpha=0.5)
	results=anvil.mpl_util.plot_image()

	print('ok')
	return results

anvil.server.wait_forever()
