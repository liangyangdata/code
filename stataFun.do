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
