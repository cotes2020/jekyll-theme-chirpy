# file.txt:
# parent pid, child pid, exe
# 110,111,chrome
# 111,222,target
# 333,444,other.exe
# 110,123,firefox
# 123,223,target

def child_pid(file):
    child_list = []
    broswer_list = ['chrome', 'firefox']
    with open(file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            p_pid = lines[i].split(",")[2].strip()
            if(p_pid in broswer_list):
                c_pid = lines[i+1].split(",")[1]
                child_list.append(c_pid)
    f = open('/Users/graceluo/Documents/GitHub/ocholuo.github.io/_posts/04CodeNote/interview/child_pid/output.txt', 'w')
    for pid in child_list:
        f.write(pid + "\n")
    f.close()
        
file = '/Users/graceluo/Documents/GitHub/ocholuo.github.io/_posts/04CodeNote/interview/child_pid/file.txt'
child_pid(file)


