# source("/Users/mac/Library/CloudStorage/OneDrive-个人/Research/05-Programming/02-R/Required.R") # 预先配置R+Python环境

# 定义要加载的包列表
packages <- c("zoo", "ggplot2", "tidyverse", "ggthemes", "formatR","readxl","data.table","data.table","showtext")

# 一次性加载所有包，并抑制启动消息
suppressPackageStartupMessages(lapply(packages, library, character.only = TRUE))
showtext_auto()

# 自定义R函数
growth_rate <- function(value) return((value - lag(value, 1))/lag(value, 1)) # 增长率计算
longer <- function(df,year) return(pivot_longer(df,-year, names_to = "key", values_to = "value")) # 变为长表
lines <- function(df,year) return(ggplot(data=longer(df,year), aes(x = year, y = value, color=key)) + geom_line())  # 画多条线图

# 定义一个函数，它接受另一个函数作为参数
apply_function <- function(data, func) {
  data %>% func()
}

# 自定义加载包函数
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg,
                     dependencies = TRUE) 
  sapply(pkg, 
         require,    
         character.only = TRUE)
}

# packages <- c("tidyverse", # Data wrangling 
#              "gapminder", # Data source
#              "knitr", # R Markdown styling 
#              "htmltools") # .html files manipulation
# ipak(packages)


# # 导入Python模块
# reticulate::use_python("/usr/local/bin/python3.7")
# reticulate::py_run_string("import sys")  #运行Python代码
# reticulate::py_run_string('sys.path.append("/Users/mac/Library/CloudStorage/OneDrive-个人/Research/05-Programming/01-Python")') #添加自定义Py函数
# myPyFun <- reticulate::import("myPyFun") # 导入我的Python常用函数

# # 导入Python包
# library(reticulate)
# os <- import("os")   # 调用函数：os$listdir(".")
# pd <- import("pandas")
# nps <- import("numpy")
# plt <- import("matplotlib.pyplot")
# py_run_string("import pandas_datareader.data as web
# import scipy.optimize as solver
# import datetime as dt
# import matplotlib.pyplot as plt
# 支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
#")


