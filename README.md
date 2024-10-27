# 常用代码片段

## Stata
```stata
**# 工作路径设置 #1
cap program drop GWD
program GWD 
args Path 
if "`Path'" == "" {
    local Path = "`c(pwd)'"  // 如果为空，使用当前工作目录
}
else {
    local Path = "$projects/`Path'"  // 否则，构建路径
}
global Project `Path'
global Data `Path'/Data
global Figure `Path'/Out/Figure
global Table `Path'/Out/Table
global Raw `Path'/Raw
global Do `Path'/Do
global Log `Path'/Log
global File `Path'/File
dis in w  "当前项目：`Path'" 
dis in w  "检查路径: " `"{stata macro list : macro list}"'
end

**# 新建项目 #2
capture program drop NWP
program NWP
args FILE    
mkdir $projects/`FILE'
mkdir $projects/`FILE'/Do
mkdir $projects/`FILE'/Raw
mkdir $projects/`FILE'/Data
mkdir $projects/`FILE'/Log
mkdir $projects/`FILE'/File
mkdir $projects/`FILE'/Out
mkdir $projects/`FILE'/Out/Table
mkdir $projects/`FILE'/Out/Figure
doe $projects/`FILE'/Do/main.do
end

**# 删除项目 #3
capture program drop RMP
program RMP
args FILE 
!rm -rf $projects/`FILE'
end

**# 启动时自动创建日志文件 #4
capture program drop auto_log
program auto_log
local fn = subinstr("`c(current_time)'",":","-",2)
local fn1 = subinstr("`c(current_date)'"," ","",3)
log    using $stlog/log-`fn1'-`fn'.log, text replace
cmdlog using $stlog/cmd-`fn1'-`fn'.log, replace
end


**# 文档写入文本内容
capture program drop Write
program Write file text
args file text
file open myfile using "`c(pwd)'/`file'.do", write
file write myfile "`text\n'"
file close myfile

```

## R 


```r
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


```




## Python

```python
#!/usr/bin/env python
"""
调包+自定义函数(wdi_get,draw_df,hp_cycle,hp_draw,hp_trend,merge_df,myggplot,pipex,reg)！

"""
import sys
sys.path.append("/Users/mac/Library/CloudStorage/OneDrive-个人/Research/05-Programming/01-Python")
# data_path = "/Users/mac/Github/gitdata"
data_path = "/Users/mac/Library/CloudStorage/OneDrive-个人/Research/02-Analysis/00-data/"
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_datareader.data as web
import wbgapi as wb
import scipy.optimize as solver
import matplotlib.pyplot as plt


# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
print("调用 My Python Tool!")

'''
import os
# os.environ['R_HOME']="C:\\Program Files (x86)\\R\\R-4.4.1" # Windows 调用R
os.environ['R_HOME']="/Library/Frameworks/R.framework/Versions/4.3-x86_64/Resources" # Mac调用R
# os.environ['R_HOME']="/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources" # Mac调用R
import rpy2.ipython
x = "`%load_ext rpy2.ipython`"
import rpy2.robjects as robjects
from rpy2.robjects import r
r('source("/Users/mac/source.R") # 预先配置R+Python环境')
print("调用R成功,jupyter中运行{}并输入%%R(快捷键：`Command+Alt+Contrl+R`）即可使用R！".format(x))
'''
# %load_ext rpy2.ipython

import stata_setup
# stata_setup.config(r"C:\Program Files\Stata18", "mp", splash=False) # Windows 调用Stata
stata_setup.config(r"/Applications/Stata", "mp", splash=False) # Mac 调用Stata
from pystata import stata
from glob import glob
from sfi import Data
print("调用Stata成功，输入%%stata（快捷键：`Command+Alt+Contrl+S`）即可使用Stata！")


# import matlab
# import matlab.engine
# eng = matlab.engine.start_matlab()

# import julia from julia.api
# import Julia
# jl = Julia(compiled_modules=False)

# 自定义函数调用方式
# import sys
# sys.path.append("/Users/mac")
# from myPyFun import hp_cycle # 导入自定义HP函数

def iso23(iso2):
    """
    将ISO2转为ISO3国家代码
    """
    import pycountry
    return pycountry.countries.get(alpha_2=iso2).alpha_3

def iso3(country,n=3):
    """
     输入字符串模糊查找国家¶ios3
    """
    import pycountry
    try:
        return pycountry.countries.search_fuzzy(country)[0].alpha_3  # 模糊查找国家¶
    except LookupError as e:
        return f"{e} not found!"

def mmerge(dfs, index_A, index_B, how='outer'):
    """
    将多个数据框按照 A,B 两个索引进行列合并
        :param dfs: 包含多个 Pandas 数据框的列表
        :param index_A: 索引 A 名称
        :param index_B: 索引 B 名称
        :return: 合并后的 Pandas 数据框
    """
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, on=[index_A, index_B], how=how)
    return result


def replace_month(date_str):
    """
    用来替换月份缩写并格式化为 YYYY-MM
    """
    month_dict = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    for abbr, num in month_dict.items():
        date_str = date_str.replace(abbr, num)
    return re.sub(r"(\d{4})(\d{2})", r"\1-\2", date_str)


def reg(df, yvar, xvars):
    """
    OLS线性回归:
    sm.OLS(Y, X, missing='drop').fit()
    --> result.summary()
    """
    import pandas as pd
    import statsmodels.api as sm
    data=pd.concat([df[yvar], df[xvars]], axis=1)
    Y = df[yvar]
    x = df[xvars]
    X = sm.add_constant(x)
    result = sm.OLS(Y, X, missing='drop').fit()
    # return yvar,result.params[1]
    print("{}~{}：系数 = {:.3f}, p值: {:.3f}".format(yvar, xvars, result.params[1], result.pvalues[1]))
    print(result.summary())


def import_data(path,sheet=0):
    """
    读取含有中文的csv格式。
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    path = Path(path)
    if path.suffix.endswith("csv"):
        return pd.read_csv(path)
    elif path.suffix.endswith("xls"):
        return pd.read_excel(path,sheet_name=sheet)
    elif path.suffix.endswith("xlsx"):
        return pd.read_excel(path,sheet_name=sheet)
    elif path.suffix.endswith("dta"):
        return pd.read_stata(path)
    else:
        print("数据格式不包含在csv、xls，xlsx和dta之中。")


def find_text(str_list, word):
    import re
    pattern = word
    matches = [s for s in str_list if re.search(pattern, s)]
    return matches


def ggplot(df,x,y,label):
    """
    ggplot2 格式Python画图
    """
    from plotnine import ggplot, aes, geom_col, geom_text,lims, position_dodge, geom_point, geom_line
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    # 指定中文字体路径
    font = FontProperties(fname='/System/Library/Fonts/Supplemental/Songti.ttc')  # macOS 的示例路径，Windows/Linux 需要相应调整
    # 设置 `matplotlib` 字体
    plt.rcParams['font.sans-serif'] = ['Songti']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    dodge_text = position_dodge(width=0.9)     #调整文本位置
    plot = (
        ggplot(df, aes(x=x,
                       y=y,
                       color=label,
                       group=label)) #类别填充颜色
        + geom_line(aes(color='factor(label)'))
        # + geom_col(position='dodge',
        #            show_legend=False)   # modified
        # + geom_text(aes(y=-.5, label=label),
        #             position=dodge_text,
        #             color='gray',  #文本颜色
        #             size=8,   #字号
        #             angle=30, #文本的角度
        #             va='top')
        # + lims(y=(-5, 60))
    )
    print(plot)

# 添加Pandas自定义names方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("names")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in
    def __call__(self):
        """
        定义一幅图画多个时间序列的函数
        参数为：df
        """
        return self._obj.columns.to_list()


# 添加Pandas自定义export_data方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("export_data")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in
    def __call__(self, name):
        """
        定义函数导出df到00-data
        参数为：df, xxx.xlsx
        格式为：xlsx,csv,dta
        """
        import pandas as pd
        import numpy as np
        from pathlib import Path

        path = Path(name)
        if path.suffix.endswith("csv"):
            return self._obj.to_csv(data_path+name, index=False)
            print("{}数据成功导出到00-data文件夹!".format(name))
        elif path.suffix.endswith("xlsx"):
            return self._obj.to_excel(data_path+name, index=False)
            print("{}数据成功导出到00-data文件夹!".format(name))
        elif path.suffix.endswith("dta"):
            return self._obj.to_stata(data_path+name, index=False)
            print("{}数据成功导出到00-data文件夹!".format(name))
        else:
            print("错误！只能是csv、xlsx和dta之一。")


# 添加Pandas自定义type2date方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("type2date")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in

    def __call__(self, n=0):
        """
        # 转换指定列为日期并作为索引
        """
        try:             # 尝试转换为日期类型
            self._obj['date'] = pd.to_datetime(self._obj.iloc[:,n], errors='ignore')
        except:
            pass
        self._obj.set_index('date', inplace=True)
        return self._obj


# 添加Pandas自定义tsline方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("tsline")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in
    def __call__(self):
        """
        定义一幅图画多个时间序列的函数
        参数为：df
        """
        if self._obj.index.name is None:
            self._obj = self._obj.type2date()
        ax = self._obj.plot(figsize=(12, 4))  # df线图
        plt.axhline(y=0, color='black', linestyle='--')  # 添加水平线 y=0
        plt.title('Time Series Plot')
        plt.ylabel('Values')
        plt.legend(title='Legend')
        plt.show()  # 显示图表

# 添加Pandas自定义find_key方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("find_key")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in
    def __call__(self, word):
        """
        通过输入字符关键字查询对应变量的序号
        """
        column_list = self._obj.columns.to_list()
        result = pd.DataFrame([(index, item) for index, item in enumerate(column_list) if word in item],
                              columns=['Index', 'Column Name'])
        return result


# 添加Pandas自定义type2numeric方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("type2numeric")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in
    def __call__(self, n=1):
        """
            2.	转换数字列：使用 pd.to_numeric() 转换为适当的数值类型。
            3.	使用 infer_objects()：让 Pandas 自动推断对象类型。
            4.	自动处理所有列：使用 apply() 结合自定义转换函数。
        """
        for col in self._obj.columns[n:]:
            # 尝试转换为数字类型
            try:
                self._obj[col] = pd.to_numeric(self._obj[col], errors='ignore')
            except:
                pass

        return self._obj


# 添加Pandas自定义export_sheet方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("export_sheet")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in
    def __call__(self, name, sheet):
        """
        定义函数导出df到00-data
        参数为：df
        格式为：xlsx/sheet
        """
        import pandas as pd
        import numpy as np
        with pd.ExcelWriter(data_path + name+".xlsx") as xlsx:
            self._obj.to_excel(xlsx, sheet_name=sheet, index=False)
        print("数据成功导出到00-data文件夹{0}/{1}表格!".format(name,sheet))


# 添加Pandas自定义select方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("select")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj  # The DataFrame being passed in
    def __call__(self,regex):
        """
        利用filter定义筛选列函数方法。
        参数：“regex”表达式
        """
        return self._obj.filter(regex=regex, axis=1)


# 批量添加Pandas自定义my函数方法
from pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor("myfun")
class CustomAccessor:
    def __init__(self, pandas_obj):
        """
        # Using the custom methods
        - df.custom.tsline(x='date')
        - df.custom.summary_statistics()
        - rolling_avg_df = df.custom.rolling_average(column='value1', window=3)
        - print(rolling_avg_df)
        """
        self._obj = pandas_obj  # The DataFrame being passed in


    def summary_statistics(self):
        """
        计算并返回DataFrame的基本统计信息
        """
        stats = self._obj.describe()
        print("Summary Statistics:")
        print(stats)
        return stats

    def rolling_average(self, column, window=3):
        """
        计算指定列的滚动平均值
        参数：column - 要计算的列名, window - 滚动窗口大小
        """
        self._obj[f'{column}_rolling'] = self._obj[column].rolling(window=window).mean()
        print(f"Rolling average for {column} with window size {window}:")
        return self._obj[[column, f'{column}_rolling']]


def penn_tri_data():
    # penn_table_variables = {
    #     'countrycode': '国家代码，通常是 ISO 3 位国家代码（如 USA 表示美国）',
    #     'country': '国家名称',
    #     'currency_unit': '货币单位（如美元、欧元等）',
    #     'year': '年份',
    #     'rgdpe': '实际国内生产总值（支出法），以 2017 年不变价格衡量',
    #     'rgdpo': '实际国内生产总值（生产法），以 2017 年不变价格衡量',
    #     'pop': '人口，国家的总人口数',
    #     'emp': '就业人数',
    #     'avh': '平均每年工作时长（小时）',
    #     'hc': '人力资本指数（基于平均受教育年限和回报率）',
    #     'ccon': '实际私人消费支出',
    #     'cda': '实际资本折旧额',
    #     'cgdpe': '实际国内生产总值（支出法）',
    #     'cgdpo': '实际国内生产总值（生产法）',
    #     'cn': '实际净资本存量',
    #     'ck': '实际资本存量',
    #     'ctfp': '总要素生产率（根据支出法）',
    #     'cwtfp': '调整后的总要素生产率（根据支出法，调整后的误差）',
    #     'rgdpna': '实际国内生产总值（国家账户法），以 2017 年不变价格衡量',
    #     'rconna': '实际私人消费（国家账户法），以 2017 年不变价格衡量',
    #     'rdana': '实际国内总资本形成（国家账户法）',
    #     'rnna': '实际净资本形成（国家账户法）',
    #     'rkna': '实际资本存量（国家账户法）',
    #     'rtfpna': '实际总要素生产率（国家账户法）',
    #     'rwtfpna': '实际加权总要素生产率（国家账户法）',
    #     'labsh': '劳动收入份额',
    #     'irr': '内部回报率（投资回报率）',
    #     'delta': '资本折旧率',
    #     'xr': '市场汇率（相对于美元）',
    #     'pl_con': '私人消费的购买力平价（PPP）',
    #     'pl_da': '总资本形成的购买力平价（PPP）',
    #     'pl_gdpo': 'GDP（生产法）的购买力平价（PPP）',
    #     'i_cig': 'CPI 和 GDP 平均值之间的指数（货币条件）',
    #     'i_xm': '出口和进口价格指数之间的平均值',
    #     'i_xr': 'CPI 和汇率之间的偏差指标',
    #     'i_outlier': '异常值的标识符（1 表示异常）',
    #     'i_irr': '投资回报率的标识符（1 表示异常）',
    #     'cor_exp': '国家账户数据的经验调整',
    #     'statcap': '国家统计能力指标',
    #     'csh_c': '消费在 GDP 中的份额',
    #     'csh_i': '投资在 GDP 中的份额',
    #     'csh_g': '政府支出在 GDP 中的份额',
    #     'csh_x': '出口在 GDP 中的份额',
    #     'csh_m': '进口在 GDP 中的份额',
    #     'csh_r': '农业在 GDP 中的份额',
    #     'pl_c': '私人消费的相对价格',
    #     'pl_i': '投资的相对价格',
    #     'pl_g': '政府支出的相对价格',
    #     'pl_x': '出口的相对价格',
    #     'pl_m': '进口的相对价格',
    #     'pl_n': '净出口的相对价格',
    #     'pl_k': '资本存量的相对价格',
    #     'code': '潘恩表的国家代码',
    #     'ERR': '汇率制度',
    #     'MI': '货币政策独立性',
    #     'OPEN': '资本自由度度'
    # }
    import pandas as pd
    import numpy as np
    df = pd.read_excel(data_path+"penn_tri.xlsx")
    return(df)



def hp_cycle(y, lamb=1600):
    """
    HP滤波并返回cycle
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle,trend = hpfilter(y, lamb)
    return cycle


def hp_trend(y, lamb=1600):
    """
    HP滤波并返回trend
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle,trend = hpfilter(y, lamb)
    return trend


def hp_draw(y, lamb=1600):
    """
    HP滤波并画图
    根据Ravn and Uhlig(2002)的建议，
    参数lambda：
        - 对于年度数据lambda参数取值6.25(1600/4^4)，
        - 对于季度数据取值1600，
        - 对于月度数据取值129600(1600*3^4)。
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle,trend = hpfilter(y, lamb)
    var_name = y.name
    # 绘制结果
    fig, ax = plt.subplots(figsize=(12, 3))
    y.plot(ax=ax, label = f'{var_name}: data')
    trend.plot(ax=ax, label=f'{var_name}: trend')
    cycle.plot(ax=ax, label=f'{var_name}: cycle')
    ax.set_xlabel("")
    ax.set_ylabel(f'{var_name}')
    ax.legend()
    plt.show()



series = {
    "BN.CAB.XOKA.GD.ZS": "经常账户差额占GDP比重（%）",
    "NE.TRD.GNFS.ZS": "贸易占GDP比重（%）"
}
economy = ["CHN","USA"]
def wdi_get(series=series,economy=economy[0],start=2000,end=2024):
    """
    从WDI获取指定国家、时间和指标的数据。
    series = {
    "BN.CAB.XOKA.GD.ZS": "经常账户差额占GDP比重（%）",
    "NE.TRD.GNFS.ZS": "贸易占GDP比重（%）"
    }
    economy = ["CHN","USA"]

    # wb.source.info()
    # wb.series.info()        # WDI by default
    # wb.series.info('NY.GDP.PCAP.CD')           # GDP
    # wb.economy.info(db=6)   # economies in the Debt Statistics database
    # wb.db = 1               # Change default database to...
    # wb.economy.coder(['Argentina', 'Swaziland', 'South Korea', 'England', 'Chicago'])
    # wb.economy.info(['CAN', 'USA', 'MEX'])     # Countries in North America
    # wb.series.info(q='flow')
    # wb.search('flow')
    # wb.economy.info(q='china')
    # wb.economy.info(wb.income.members('HIC'))      # high-income economies
    # wb.series.info(wb.topic.members(8))            # indicators in the health topic (wb.topic.info() for full list)
    # wb.series.info(topic=8)                        # same as above but easier to type
    # wbopendata, country(ago;bdi;chi;dnk;esp) indicator(sp.pop.0610.fe.un) year(2000:2010) clear  long
    # wb.data.DataFrame(['NY.GDP.PCAP.CD', 'SP.POP.TOTL'], 'CAN', mrv=5) # most recent 5 years
    # wb.data.DataFrame('SP.POP.TOTL', wb.region.members('AFR'), range(2010, 2020, 2))
    """
    import pandas as pd
    import numpy as np
    import wbgapi as wb
    data = wb.data.DataFrame(
        series,
        economy,
        time=range(start, end+1),
        skipBlanks=True,  # 跳过空白值
        columns="series"  # 按系列排列
    )
    data = data.rename(columns=series).reset_index()
    data["year"] = data.time.str.replace("YR","").to_list()
    return(data)
# wdi_get().head()
```
