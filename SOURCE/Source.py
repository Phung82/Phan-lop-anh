#Import các thư viện hỗ trợ

import numpy as np
from tkinter import *
from tkinter import filedialog as fd
import PIL
from PIL import Image, ImageTk, ImageEnhance
import CNN
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
import shutil
import fk
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

class Application(Frame):
    def __init__(self):
        super().__init__()
        self.master.title("Đồ Án Môn Học")
        self.pack()

        menubar = Menu(self.master)
        #menu
        filemenu = Menu(menubar, tearoff = 0)
        filemenu.add_command(label= "Open", command=self.Open_file)
        filemenu.add_command(label= "Connect", command=self.Choose_CNN)
        filemenu.add_separator()
        filemenu.add_command(label= "Exit", command=self.master.destroy)
        menubar.add_cascade(label= "Chức năng", menu=filemenu)

        controlmenu = Menu(menubar, tearoff=0)
        controlmenu.add_command(label = "Load dataset",command=self.Choose_dataset)
        controlmenu.add_command(label = "Mô hình mạng",command=self.Train_dts)
        controlmenu.add_command(label = "Hàm mất mát",command=self.Train_dts)
        controlmenu.add_command(label = "Huấn luyện mạng",command=self.Train_dts)
        controlmenu.add_command(label = "Kiểm tra mạng mạng",command=self.Train_dts)
        menubar.add_cascade(label = "Điều khiển", menu=controlmenu)

        aboutmenu = Menu(menubar, tearoff=0)
        aboutmenu.add_command(label = "Thông tin",command=self.About)
        aboutmenu.add_command(label = "Hướng dẫn",command=self.hdsd)
        menubar.add_cascade(label= "Giới thiệu", menu = aboutmenu)
        self.master.config(menu = menubar)
        #top
        #chia frame
        frame = Frame(self.master)
        frame.pack()
        self.label_tieude = Label(frame,pady=20, text="ỨNG DỤNG PHÂN LỚP ẢNH",fg="Blue",font=("arial", 24,"bold"),padx=20)
        self.label_tieude.pack()
        
        #middel frame
        left = Frame(self.master, borderwidth=2, relief="solid")
        right = Frame(self.master, borderwidth=2, relief="solid")
        container = Frame(left, borderwidth=2, relief="solid")
        self.fr_button = Frame(left, borderwidth=2, relief="solid")
        self.box_left = Frame(right, borderwidth=2, relief="solid")
         
        label3 = Label(self.fr_button, text=' Các tùy chọn',padx=50,height="1",font=("arial", 12))
        label3.pack()

        left.pack(side="left", expand=True, fill="both")
        right.pack(side="right", expand=True, fill="both")
        container.pack(expand=True, padx=1, pady=1)
        self.fr_button.pack(expand=True, fill="both", padx=1, pady=1)
        self.box_left.pack(expand=True, fill="both", padx=1, pady=1)

        
        #hiển thị ảnh nền
        self.img_fn = Image.open("./bin/background.jpg")
        #reimg=self.img.resize((1000,600),Image.ANTIALIAS)
        self.tatras_fn = ImageTk.PhotoImage(self.img_fn)        
        canvas_fn = Canvas(container, width=693, height=416)
        canvas_fn.create_image(1, 1, anchor=NW, image=self.tatras_fn)
        canvas_fn.pack()
        #hiển thị logo bìa
        self.img_lg = Image.open("./bin/logo.png")
        reimg=self.img_lg.resize((100,100),Image.ANTIALIAS)
        self.tatras_lg = ImageTk.PhotoImage(reimg)        
        canvas_lg = Canvas(self.box_left, width=220, height=130)
        canvas_lg.create_image(80, 10,anchor=NW, image=self.tatras_lg)
        canvas_lg.pack(fill=Y)

        #mục bìa
        Label(self.box_left, text="    ĐỒ ÁN MÔN HỌC",height="1",fg="Blue",font=("arial", 18,"bold"),pady=50).pack()
        Label(self.box_left, text="ĐỀ TÀI: TÌM HIỂU NGÔN NGỮ LẬP TRÌNH PYTHON VÀ THƯ VIỆN PYTORCH",height="1",fg="Red",font=("arial", 14),pady=5).pack()
        Label(self.box_left, text="VIẾT ỨNG DỤNG PHÂN LỚP ẢNH (BIRD, CAT, FROG,HORSE)",height="1",fg="Red",font=("arial", 14),pady=20).pack()
        Label(self.box_left, text="  Nhóm Thực Hiện",height="1",fg="Blue",font=("arial", 14),pady=20).pack()
        Label(self.box_left,anchor="w", text=" Giáo viên hướng dẫn:",height="1",font=("arial", 14)).pack(fill=X)
        Label(self.box_left,anchor="w", text=' Thạc sĩ    Ngô Thanh Tú',padx=50,height="1",font=("arial", 14)).pack(fill=X)
        Label(self.box_left,anchor="w", text=" Thành viên thực hiện:",height="1",font=("arial", 14),pady=10).pack(fill=X)
        Label(self.box_left,anchor="w", text=' Nguyễn Tiểu Phụng       17DDS0703132',padx=50,height="1",font=("arial", 14)).pack(fill=X)
        Label(self.box_left,anchor="w", text=' Huỳnh Đức Anh Tuấn    17DDS0703143',padx=50,height="1",font=("arial", 14)).pack(fill=X)

        #button
        
        bottomframe = Frame(self.fr_button,pady=20, borderwidth=2, relief="solid")
        bottomframe.pack()
        #tạo button để mở chọn ảnh - gọi lại hàm Open_file()
        bottomframe_1 = Frame(bottomframe,padx=20, borderwidth=2, relief="solid")
        bottomframe_1.pack(side = LEFT)
        self.process_btn = Button(bottomframe_1, text = "Chọn hình ảnh", fg = "black",bg='#0caffc',padx=20,command=self.Open_file)
        self.process_btn.pack( )
        #tao button de chon mo hinh mang noron
        bottomframe_2 = Frame(bottomframe,padx=20, borderwidth=2, relief="solid")
        bottomframe_2.pack(side = LEFT)
        self.CNN_btn = Button(bottomframe_2, text = "Chọn mô hình", fg = "black",bg='#F7F2E0',padx=20,command=self.Choose_CNN)
        self.CNN_btn.pack()
        
        #tạo button để test
        bottomframe_3 = Frame(bottomframe,padx=20, borderwidth=2, relief="solid")
        bottomframe_3.pack(side = LEFT)
        self.choose_btn = Button(bottomframe_3, text = "Kiểm tra",state=DISABLED, fg = "black",bg='#F7F2E0',padx=20,command=self.Test_img)
        self.choose_btn.pack()
        #tạo button để thoát
        bottomframe_4 = Frame(bottomframe,padx=20, borderwidth=2, relief="solid")
        bottomframe_4.pack(side = LEFT)
        self.exit_btn = Button(bottomframe_4, text = "Thoát", fg="black",bg='#CEF6EC',padx=20,command=self.master.destroy)
        self.exit_btn.pack()


    #Hàm lấy đường dẫn liên kết
    def Open_file(self):
        try:
            File=fd.askopenfilename(title="Open",filetype=[("file .png","*.png"),("file .jpg","*.jpg"),("All files","*")])
            self.img_bathname=File
            #xử lý đường dẫn file bằng cawsch loại bỏ "/"
            self.path=self.img_bathname.split('/')
            #lấy tên file
            self.name_img=self.path[len( self.path)-1]
        except:
            #Xử lý lỗi không tìm thấy đường dẫn
            self.path_link = "Không tìm thấy đường dẫn - Chọn lại đường dẫn!"
        label2 = Label(self.fr_button, text=self.path[-1])
        self.CNN_btn["state"]=NORMAL
        self.CNN_btn["bg"]="#CEF6EC"
        #self.process_btn["bg"]="#00ff1e"
        label2.pack()


#hàm chọn mo hình
    def Choose_CNN(self):
        try:
            File=fd.askopenfilename(title="Open",filetype=[("file .pth","*.pth"),("file .txt","*.txt"),("All files","*")])
            self.file_bathname=File
            #xử lý đường dẫn file bằng cawsch loại bỏ "/"
            self.path2=self.file_bathname.split('/')
            #lấy tên file
            self.name_file=self.path[len( self.path)-1]
        except:
            #Xử lý lỗi không tìm thấy đường dẫn
            self.path_link2 = "Không tìm thấy đường dẫn - Chọn lại đường dẫn!"
        label2 = Label(self.fr_button, text=self.path2[-1])
        self.choose_btn["state"]=NORMAL
        self.choose_btn["bg"]="#CEF6EC"
        #self.process_btn["bg"]="#00ff1e"
        label2.pack()


        #chọn ảnh để kiểm tra
    def Test_img(self):
       box_left = Toplevel(self.master)
       box_left.title("Test-img")
       bottomframe1 = Frame(box_left,pady=20, borderwidth=0, relief="solid")
       bottomframe1.pack(expand=True, fill="both", padx=1, pady=1)
       
       bottomframe2 = Frame(box_left, borderwidth=2, relief="solid")
       bottomframe2.pack(expand=True,  padx=1, pady=1)
       
       bottomframe3 = Frame(box_left,pady=10, borderwidth=0, relief="solid")
       bottomframe3.pack(expand=True, fill="both", padx=1, pady=1)
       device = CNN.get_default_device()
       
       kq=CNN.file_to_int(self.img_bathname)
       
       data_dir = './data/cifar10'
       Label(bottomframe1,fg="red", text="KẾT QUẢ NHẬN DẠNG",height="1",font=("arial", 17),padx=170).pack() 
       model2 = CNN.to_device(CNN.Cifar10CnnModel(), device)
       model2.load_state_dict(torch.load(self.file_bathname))
       test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())
       '''KT lại'''
       img, label = test_dataset[kq]
       plt.imshow(img.permute(1, 2, 0)) 
       print( ' Predicted:', CNN.predict_image(img, model2))

       #s='/test/cat/0972.png'
       self.img_test = Image.open(self.img_bathname)
       reimg=self.img_test.resize((130,130),Image.ANTIALIAS)
       self.tatras_test = ImageTk.PhotoImage(reimg)        
       canvas_lg = Canvas(bottomframe2, width=150, height=150)
       canvas_lg.create_image(80, 80,anchor=CENTER, image=self.tatras_test)
       canvas_lg.pack()
       nhan=str(CNN.predict_image(img, model2))
       Label(bottomframe3,fg="blue", text="Đây là: "+nhan,height="1",font=("arial", 17),padx=170).pack()
       box_left.mainloop()

    def Choose_dataset(self):
        File=fd.askdirectory(title="Choose")
        self.path_dir=File
        print("Đường dẫn: ",self.path_dir)
        labeldir= Label(self.box_left0, text=self.path_dir)
        labeldir.pack()

    def luuanh(self):
        file = fd.asksaveasfilename(title = "Lưu ảnh",filetype = (("png files","*.png"),("jpeg files","*.jpg"),("All files","*.*")))
        if file:
            originalImage = cv2.imread(self.path)
            #self.save_img.save(file)
            cv2.imwrite(file,originalImage)  
    def id_plot(self):
        Label(self.box_right1, text="Traning...!",height="1",fg="#01DF01",font=("arial", 14),pady=20).pack(side="left")
        self.n_echo=self.Num_echo.get()
        self.n_lr=self.Num_lr.get()
        self.n_path=self.Path_n.get()
        print("self.n_echo: ",int(self.n_echo))
        print("self.n_lr: ",self.n_lr)
        print("self.n_path: ",self.n_path)
        
    def Train_data(self):
       box_main = Toplevel(self.master)
       box_main.title("Huấn luyện mạng")
       
       Top_F = Frame(box_main, borderwidth=1, relief="solid")
       Top_F.pack(side="top", expand=True, fill="both")
       
       Bot_F = Frame(box_main, borderwidth=1, relief="solid")
       Bot_F.pack(side="bottom", expand=True, fill="both")
       
       left = Frame(Bot_F, borderwidth=1, relief="solid")
       left.pack(side="left", expand=True, fill="both")
       
       right = Frame(Bot_F, borderwidth=1, relief="solid")
       right.pack(side="right", expand=True, fill="both")
       
       self.box_left0 = Frame(left, relief="solid")
       self.box_left0.pack(side="top", expand=True, fill="both")
       self.box_left1 = Frame(left,  relief="solid")
       self.box_left1.pack(side="top", expand=True, fill="both")
       self.box_left2 = Frame(left,  relief="solid")
       self.box_left2.pack(side="top", expand=True, fill="both")
       self.box_left3 = Frame(left,  relief="solid")
       self.box_left3.pack(side="top", expand=True, fill="both")
       box_left4 = Frame(left,  relief="solid")
       box_left4.pack(side="top", expand=True, fill="both")       
       
       box_right0 = Frame(right, relief="solid")
       box_right0.pack(side="top", expand=True, fill="both")
       self.box_right1 = Frame(right, relief="solid")
       self.box_right1.pack(side="top", expand=True, fill="both")
       self.box_right2 = Frame(right, relief="solid")
       self.box_right2.pack(side="top", expand=True, fill="both")
       
       Label(Top_F,fg="red", text="HIỆU CHỈNH THÔNG SỐ",height="1",font=("arial", 17),padx=200,pady=10).pack(fill=X)
       
       choosedts_btn = Button(self.box_left0, text = "Chọn dataset",fg = "black",bg='#F7F2E0',command=self.Choose_dataset)
       choosedts_btn.pack(side="left")
       
       Label(self.box_left1, text="Num_echo:",height="1",fg="Blue",font=("arial", 14),pady=20).pack(side="left")
       
       self.Num_echo = Entry(self.box_left1,width=20,justify=CENTER)
       self.Num_echo.pack(side="left")
       #print('Num_echo:',Num_in)
       
       Label(self.box_left2, text="lr:",height="1",fg="Blue",font=("arial", 14),pady=20).pack(side="left")
       
       self.Num_lr = Entry(self.box_left2,width=20,justify=CENTER)
       self.Num_lr.pack(side="left")
       
       Label(self.box_left3, text="Path_name:",height="1",fg="Blue",font=("arial", 14),pady=20).pack(side="left")
       
       self.Path_n = Entry(self.box_left3,width=50,justify=CENTER)
       self.Path_n.pack(side="left")
       
       Label(box_right0, text="KẾT QUẢ",height="1",fg="Blue",font=("arial", 14),pady=5,padx=50).pack(fill=X)
       '''----------------------Xu ly---------------------------------'''
       device = CNN.get_default_device()
       #print(device)
       #ketnoi()
       data_dir =self.path_dir
       #print(os.listdir(data_dir))
       classes = os.listdir(data_dir)
       #print(classes)
       dataset = ImageFolder(data_dir, transform=ToTensor())
       img, label = dataset[0]
       #print(img.shape, label)
       img
       #print('dataset.classes: ',dataset.classes)
       #show_example(*dataset[0])
       #dao taao va xac thuc
       random_seed = 42 
       torch.manual_seed(random_seed);
       val_size = 5000
       train_size = len(dataset) - val_size
    
       train_ds, val_ds = random_split(dataset, [train_size, val_size])
       #print("len(train_ds), len(val_ds): ",len(train_ds), len(val_ds),)
       #print("Run 0!")
       batch_size=128    
       train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
       val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

       #show_batch(train_dl)
    
       #train model doane
       model = CNN.Cifar10CnnModel()
       model
        
       train_dl = CNN.DeviceDataLoader(train_dl, device)
       val_dl = CNN.DeviceDataLoader(val_dl, device)
       CNN.to_device(model, device);
       #print("Run 1!")
       model = CNN.to_device(Cifar10CnnModel(), device)
       CNN.evaluate(model, val_dl)
       #print("Run 2!")
    
       #num_epchs là số lần đưa dữ liệu vào mạng
       #Sử dụng gói tối ưu để xác định Trình tối ưu hóa sẽ cập nhật trọng số
       #của mô hình mạng 
       #Dùng hàm Adam cho trình tối ưu hóa các Tensors (Tensors Là dữ liệu nhiều chiều tương tự như ma trận trong numpy)
       #lr là tỷ lệ học của mô hình
       num_epochs = self.n_echo
       opt_func = torch.optim.Adam
       lr = self.n_lr

       history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    
       #print("Run 3!")
       CNN.plot_accuracies(history)
       CNN.plot_losses(history)
       #print("Run 4!")
       #luu mo hinh
       PATH = self.n_path
       torch.save(model.state_dict(), PATH)
       choosedts_run = Button(box_left4, text = "HUẤN LUYỆN",fg = "black",bg='#F7F2E0',pady=10,command=self.Show_plot)
       choosedts_run.pack(fill=X)
       #print("Show roi!")
       box_main.mainloop()

    def About(self):
       self.box_left = Toplevel(self.master)
       self.box_left.title("About")
       Label(self.box_left,fg="red", text="HỌC PHẦN MẠNG NUERAL",height="1",font=("arial", 17),padx=200).pack()
       Label(self.box_left, text="    ĐỒ ÁN MÔN HỌC",height="1",fg="Blue",font=("arial", 18,"bold"),pady=20).pack()
       Label(self.box_left, text="ĐỀ TÀI: TÌM HIỂU THƯ VIỆN PYTORCH, NGÔN NGỮ LẬP TRÌNH PYTHON",height="1",fg="Red",font=("arial", 14),pady=5).pack()
       Label(self.box_left, text="VIẾT ỨNG DỤNG PHÂN LỚP ẢNH",height="1",fg="Red",font=("arial", 14),pady=20).pack()
       Label(self.box_left, text="  Nhóm Thực Hiện",height="1",fg="Blue",font=("arial", 14),pady=20).pack()
       Label(self.box_left,anchor="w", text=" Giáo viên hướng dẫn:",height="1",font=("arial", 14)).pack(fill=X)
       Label(self.box_left,anchor="w", text=' Thạc sĩ    Ngô Thanh Tú',padx=50,height="1",font=("arial", 14)).pack(fill=X)
       Label(self.box_left,anchor="w", text=" Thành viên thực hiện:",height="1",font=("arial", 14),pady=10).pack(fill=X)
       Label(self.box_left,anchor="w", text=' Nguyễn Tiểu Phụng       17DDS0703132',padx=50,height="1",font=("arial", 14)).pack(fill=X)
       Label(self.box_left,anchor="w", text=' Huỳnh Đức Anh Tuấn    17DDS0703143',padx=50,height="1",font=("arial", 14)).pack(fill=X)
       self.box_left.mainloop()

    def luuaanh(self):
        file = fd.asksaveasfilename(title = "Lưu ảnh",filetype = (("png files","*.png"),("jpeg files","*.jpg"),("All files","*.*")))
        shutil.copyfile(self.path, file)
 
       
    def Show_plot(self):
        #Label(self.box_right1, text="Traning...!",height="1",fg="#01DF01",font=("arial", 14),pady=20).pack(side="left")
        self.n_echo=self.Num_echo.get()
        self.n_lr=self.Num_lr.get()
        self.n_path=self.Path_n.get()
        print("self.n_echo: ",int(self.n_echo))
        print("self.n_lr: ",self.n_lr)
        print("self.n_path: ",self.n_path)
        
        t=int(self.n_echo)
        if fk.countdown(t)==0:
            Label(self.box_right1, text="Done!",height="1",fg="Blue",font=("arial", 14),pady=20).pack(side="left")
            self.path="./bin/pycache/img/"+str(t)+".png"
            self.img_plot = Image.open(self.path) 
            self.reipl=self.img_plot.resize((386,278),Image.ANTIALIAS)
            self.tatras_plot = ImageTk.PhotoImage(self.reipl)        
            canvas_pl = Canvas(self.box_right2, width=400, height=300)
            canvas_pl.create_image(200, 160,anchor=CENTER, image=self.tatras_plot)
            canvas_pl.pack()
            print("Da show!:",self.path)
        fk.rename_f(t)

        
    def Train_dts(self):
       box_main = Toplevel(self.master)
       box_main.title("Huấn luyện mạng")
       
       Top_F = Frame(box_main, borderwidth=1, relief="solid")
       Top_F.pack(side="top", expand=True, fill="both")
       
       Bot_F = Frame(box_main, borderwidth=1, relief="solid")
       Bot_F.pack(side="bottom", expand=True, fill="both")
       
       left = Frame(Bot_F, borderwidth=1, relief="solid")
       left.pack(side="left", expand=True, fill="both")
       
       right = Frame(Bot_F, borderwidth=1, relief="solid")
       right.pack(side="right", expand=True, fill="both")
       
       self.box_left0 = Frame(left, relief="solid")
       self.box_left0.pack(side="top", expand=True, fill="both")
       self.box_left1 = Frame(left,  relief="solid")
       self.box_left1.pack(side="top", expand=True, fill="both")
       self.box_left2 = Frame(left,  relief="solid")
       self.box_left2.pack(side="top", expand=True, fill="both")
       self.box_left3 = Frame(left,  relief="solid")
       self.box_left3.pack(side="top", expand=True, fill="both")
       box_left4 = Frame(left,  relief="solid")
       box_left4.pack(side="top", expand=True, fill="both")       
       
       box_right0 = Frame(right, relief="solid")
       box_right0.pack(side="top", expand=True, fill="both")
       self.box_right1 = Frame(right, relief="solid")
       self.box_right1.pack(side="top", expand=True, fill="both")
       self.box_right2 = Frame(right, relief="solid")
       self.box_right2.pack(side="top", expand=True, fill="both")
       self.box_right3 = Frame(right, relief="solid")
       self.box_right3.pack(side="top", expand=True, fill="both")
       
       Label(Top_F,fg="red", text="HIỆU CHỈNH THÔNG SỐ",height="1",font=("arial", 17),padx=200,pady=10).pack(fill=X)
       
       choosedts_btn = Button(self.box_left0, text = "Chọn dataset",fg = "black",bg='#F7F2E0',command=self.Choose_dataset)
       choosedts_btn.pack(side="left")
       
       Label(self.box_left1, text="Num_echo:",height="1",fg="Blue",font=("arial", 14),pady=20).pack(side="left")
       
       self.Num_echo = Entry(self.box_left1,width=20,justify=CENTER)
       self.Num_echo.pack(side="left")
       #print('Num_echo:',Num_in)
       
       Label(self.box_left2, text="lr:",height="1",fg="Blue",font=("arial", 14),pady=20).pack(side="left")
       
       self.Num_lr = Entry(self.box_left2,width=20,justify=CENTER)
       self.Num_lr.pack(side="left")
       
       Label(self.box_left3, text="Path_name:",height="1",fg="Blue",font=("arial", 14),pady=20).pack(side="left")
       
       self.Path_n = Entry(self.box_left3,width=50,justify=CENTER)
       self.Path_n.pack(side="left")
       
       Label(box_right0, text="KẾT QUẢ",height="1",fg="Blue",font=("arial", 14),pady=5,padx=50).pack(fill=X)
       
       choosedts_run = Button(box_left4, text = "HUẤN LUYỆN",fg = "black",bg='#F7F2E0',pady=10,command=self.Show_plot)
       choosedts_run.pack(side="bottom",fill=X)
       choosedts_run = Button(self.box_right3, text = "LƯU ĐỒ THỊ",fg = "black",bg='#F7F2E0',pady=10,command=self.luuaanh)
       choosedts_run.pack(side="bottom",fill=X)
       #print("Show roi!")
       box_main.mainloop()
    #cửa sổ hướng dẫn sử dụng
    def hdsd(self):
       filewin = Toplevel(self.master)
       filewin.title("Hướng dẫn sử dụng")
       Label(filewin,fg="red", text="HƯỚNG DẪN SỬ DỤNG PHẦN MỀM",height="1",font=("arial", 17),padx=200,pady=20).pack()
       Label(filewin,anchor="w", text="- Bước 1: Chọn hình ản",height="1",font=("arial", 14)).pack(fill=X)
       Label(filewin,anchor="w", text='+ Cách 1: Click vào button "Hình ảnh"',padx=30,height="1",font=("arial", 14)).pack(fill=X)
       Label(filewin,anchor="w", text='+ Cách 2: Chức năng -> Open',padx=30,height="1",font=("arial", 14)).pack(fill=X)
       Label(filewin,anchor="w", text="- Bước 2: Chọn mô hình",height="1",font=("arial", 14)).pack(fill=X)
       Label(filewin,anchor="w", text='+ Cách 1: Click vào button "Chọn mô hình"',padx=30,height="1",font=("arial", 14)).pack(fill=X)
       Label(filewin,anchor="w", text='+ Cách 2: Chức năng -> Connect CNN"',padx=30,height="1",font=("arial", 14)).pack(fill=X)
       Label(filewin,anchor="w", text="- Bước 3: Kiểm tra",height="1",font=("arial", 14)).pack(fill=X)
       filewin.mainloop()  
#hàm main
def main():
    window = Tk()
    window.geometry("1400x700")
    app = Application()
    window.mainloop()
   


#gọi lại hàm main()
if __name__ == '__main__':
    main()     
