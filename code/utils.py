import numpy as np
import pandas as pd


def caculate_edit_distance(a,b): # 计算字符串的编辑距离
    m = len(a)
    n = len(b)
    if m == 0 or n == 0:
        return m if n == 0 else n

    a = "#" + a
    b = "#" + b
    dp = [[0]*(n+1) for _ in range(m+1)] # m * n,dp[i][j]表示a中前i个字符和b中前j个字符的编辑距离
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = min(min(dp[i-1][j],dp[i][j-1]) + 1,dp[i-1][j-1] if a[i] == b[j] else dp[i-1][j-1] + 1)
    return dp[m][n]
def construct_features(AuthorId,PaperId,Entity):
    # 1. 作者发过的论文总数
    paper_list = Entity.Author[AuthorId]['Paper'] # 获取该作者的所有论文的 paperId
    total_paper_num_of_author = len(paper_list)

    # 2. 该篇论文的作者总数

    author_list = Entity.Paper[PaperId]['Author']  # 获取该篇论文的所有作者的信息，包括id，name，Affiliation
    total_author_num = len(author_list)

    num_confAndjour = 0

    # 3. 提取该作者在该会议或期刊上的论文总数
    conferenceId = Entity.Paper[PaperId]['ConferenceId']  #  conferenceId 和 journalId一定只有一个非0
    journalId = Entity.Paper[PaperId]['JournalId']

    for paper in paper_list:
        try:
            if Entity.Paper[paper].get('ConferenceId') != None and Entity.Paper[paper]['ConferenceId'] == conferenceId:
                num_confAndjour += 1
            elif Entity.Paper[paper].get('JournalId') != None and    Entity.Paper[paper]['JournalId'] == journalId:
                num_confAndjour += 1
        except:
            print(paper)
            print(Entity.Paper[paper])
    # 4. 作者与该论文的其它作者的合作总次数
    coop_num_author = 0
    for paper in paper_list:
        current_paper_author = Entity.Paper[paper]['Author']   # 当前所遍历的paper的所有作者
        for author_info in author_list:
            if author_info['Id'] != AuthorId  and author_info['Id'] in current_paper_author:
                coop_num_author += 1

    # 5. 计数count times <authorid, paperid> appear in PaperAuthor.csv
    author_paper_count = 0
    for paper in paper_list:
        if paper == PaperId:
            author_paper_count += 1



    # 6. 计算author对应的name和affiliation 与 PaperAuthor.csv中出现的name 和 affiliation的 edit distance
    author_name  = Entity.Author[AuthorId]['Name']
    if author_name == None:
        author_name = ""
    author_affiliation = Entity.Author[AuthorId]['Affiliation']
    if author_affiliation == None:
        author_affiliation = ""
    Name_list_in_PaperAuthor = []
    Affiliation_list_in_PaperAuthor = []
    for author_info in author_list:
        if author_info['Id'] == AuthorId:
            if author_info['Name'] != None and author_info['Affiliation'] != None:
                Name_list_in_PaperAuthor.append(author_info['Name'])
                Affiliation_list_in_PaperAuthor.append(author_info['Affiliation'])
    name_affiliation_edit_distance = 0
    if len(Name_list_in_PaperAuthor) == 0:
        name_affiliation_edit_distance = len(author_name) + len(author_affiliation)
    else:
        for name,affiliation in zip(Name_list_in_PaperAuthor,Affiliation_list_in_PaperAuthor):

            name_affiliation_edit_distance += caculate_edit_distance(name,author_name) + caculate_edit_distance(affiliation,author_affiliation)
        name_affiliation_edit_distance /= len(Name_list_in_PaperAuthor)

    # 7. 该作者的所有论文的发布时间的平均值 与 该篇论文发布时间的差值
    average_year_diff = -1
    time_list = []
    this_paper_year = 0
    if  Entity.Paper[PaperId].get('Year') != None:
        this_paper_year = Entity.Paper[PaperId].get('Year')
    for paper in paper_list:
        if Entity.Paper[paper].get('Year') != None:
            time_list.append(Entity.Paper[paper].get('Year'))
    if len(time_list) != 0 :
        average_year_diff = sum(time_list) / len(time_list) - this_paper_year
    return [total_paper_num_of_author,total_author_num,num_confAndjour,coop_num_author,author_paper_count,name_affiliation_edit_distance,average_year_diff]



if __name__ == "__main__":
    ret = caculate_edit_distance("baacd","aaad")
    print(ret)

