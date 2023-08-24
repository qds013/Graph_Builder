import pandas as pd
import matplotlib.pyplot as plt
import math
import PySimpleGUI as sg
from datetime import datetime, timedelta
from fpdf.enums import XPos, YPos
from fpdf import FPDF
from pathlib import Path
from chardet import detect
import os
from statistics import mean
import csv
import sys
import warnings
warnings.filterwarnings("ignore")   
p = os.path.dirname(__file__)
Path(f"{p}/results").mkdir(parents=True, exist_ok=True)
def diametr(h,V):
    return round((math.sqrt((abs((4*V)/(math.pi*h))))*100),0)
def Invoker(filename):
    GuhHeader1 = ['GHSYS', '<>', '10094401', 'UFA M7', 'ym Geoizol', 'AVTOBAN', 'FUNDEX', '0', '0', '800', '0', '4',
                '4514AD072']
    GuhHeader2 = 'Date;Depth;Amperage;P_Air;Weight;Load;Buckets;Volume;X_Axis;Y_Axis;Zone;Level;P_Water'
    GuhHeader3 = '0;m;A;bar;to;to;0;m³;°;°;0;%;bar'
    FileList = [filename]
    '''if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            FileList.append(sys.argv[i])
            i += 1
    else:
        print("No files specified - looking for <*.csv> files in current directory...")
        for f in os.listdir():
            if f.endswith(".csv"):
                FileList.append(f)'''
    if len(FileList) > 0:
        # print(FileList)
        for f in FileList:
            c = os.path.splitext(f)[0]
            guh = c + '.guh'
            c=str(c.split('/')[-1])
            with open(f, encoding='cp1252') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                with open(guh, 'w', encoding='cp1252') as guh_file:
                    guh_file.write(
                        "{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12}\n".format(GuhHeader1[0], c, GuhHeader1[2],
                                                                                        GuhHeader1[3], GuhHeader1[4],
                                                                                        GuhHeader1[5], GuhHeader1[6],
                                                                                        GuhHeader1[7], GuhHeader1[8],
                                                                                        GuhHeader1[9], GuhHeader1[10],
                                                                                        GuhHeader1[11], GuhHeader1[12]))
                    guh_file.write("{}\n".format(GuhHeader2))
                    guh_file.write("{}\n".format(GuhHeader3))
                    count = 1
                    for row in csv_reader:
                        if count > 2:
                            guh_file.write(
                                "{0};{1:0.2f};{2};{3:0.1f};{4:0.2f};{5};{6};{7:0.2f};{8:0.1f};{9:0.1f};{10};{11};{12}\n".format(
                                    row[2], float(row[3]), int(row[4]), float(row[5]), float(row[6]), int(row[7]),
                                    int(row[8]), int(float(row[9])), float(row[10]), float(row[11]), row[12], row[13], row[14]))
                        count += 1
                    guh_file.close
                csv_file.close
def CRISTAL_MAIDEN(filename):
    mas = []
    max_mas = []
    bucket = set()
    Miracle = {}
    max_high = []
    Mean_Amperage = []
    MAX_Amperage = []
    diametr_mas = []
    Height = []
    Time_F = []
    V_bucket = 1.93
    Finally = [
        ['Бакет[No]', 'Начало работы бакета[m]', 'Конец работы бакета[m]', 'Высота бакета[m]', 'Время работы бакета',
        'Ср.значение амперметра[A]', 'Макс. знач. Ампер[A]', 'Диаметр','Ударов за метр']]
    Depth={}
    with open(filename,'r') as f:
        f.readline()
        f.readline()
        f.readline()
        for line in f:
            line = line.split(';')
            mas.append(line)
    for line in range((len(mas) - 1), -1, -1):
        if int(float(mas[line][6])) == 0:
            del mas[line]
        else:
            bucket.add(int(float((mas[line][6]))))
    for i in bucket:
        Miracle[i] = [0]
        Depth[i]=[]
    for line in range((len(mas) - 1), 0, -1):
        if (mas[line][1]) == (mas[line - 1][1]) or (mas[line][6] != mas[line - 1][6]):
            pass
        else:
            Miracle[int(float((mas[line][6])))][0] += 1
            Depth[int(float((mas[line][6])))].append((mas[line][1]))
    for i in bucket:
        while int(mas[line][6]) == i and line < (len(mas) - 1):
            max_mas.append(int(mas[line][2]))
            max_high.append((mas[line][1]))
            line += 1
        Miracle[i].append(round(mean(max_mas), 2))
        Miracle[i].append(max(max_mas))
        Miracle[i].append(max_high[0])
        Miracle[i].append(max_high[-1])
        Miracle[i].append(round(abs(float(max_high[-1]) - float(max_high[0])), 2))
        Miracle[i].append(diametr(float(Miracle[i][5]), V_bucket))
        max_mas = []
        max_high = []
        axe = []
    for i in Miracle.values():
        Time_F.append(i[0])
        Mean_Amperage.append(i[1])
        MAX_Amperage.append(i[2])
        Height.append(i[5])
        diametr_mas.append(i[6])
    for i in Miracle:
        min_ts = Miracle[i][0] // 60
        min_pl = Miracle[i][0] % 60
        Miracle[i][0] = '00:' + str(f"{min_ts:02d}") + ':' + str(f"{min_pl:02d}")
    for i in Miracle:
        Finally.append([str(i)])
        Finally[i].append(str(Miracle[i][3]))
        Finally[i].append(str(Miracle[i][4]))
        Finally[i].append(str(Miracle[i][5]))
        Finally[i].append(str(Miracle[i][0]))
        Finally[i].append(str(Miracle[i][1]))
        Finally[i].append(str(Miracle[i][2]))
        Finally[i].append(str(Miracle[i][6]))
    time = '00:' + str(f"{(min(Time_F)) // 60:02d}") + ':' + str(f"{(min(Time_F)) % 60:02d}")
    Time = '00:' + str(f"{(max(Time_F)) // 60:02d}") + ':' + str(f"{(max(Time_F)) % 60:02d}")
    for i in Depth:
        stokes = 0
        for j in range(2,len(Depth[i])-2):
            if Depth[i][j]>Depth[i][j+1] and Depth[i][j+1]>Depth[i][j+2] and  Depth[i][j]>Depth[i][j-1] and  Depth[i][j-1]>Depth[i][j-2]:
                stokes+=1
        Finally[i].append(str(round((stokes/float(Finally[i][3])),2)))
    for i in range(1,len(Finally)):
        axe.append(float(Finally[i][8]))
    Finally.append(['MIN:', '', '', str(min(Height)), str(time), str(min(Mean_Amperage)), str(min(MAX_Amperage)),
                    str(min(diametr_mas)),str(min(axe))])
    Finally.append(['MAX:', '', '', str(max(Height)), str(Time), str(max(Mean_Amperage)), str(max(MAX_Amperage)),
                    str(max(diametr_mas)),str(max(axe))])
    return Finally
class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def footer(self):
        self.set_y(-15)
        self.add_font("NotoSans", style="", fname="fonts/NotoSans-Regular.ttf")
        self.set_font("NotoSans", "", 8)
        self.cell(
            0,
            8,
            "Пушкинский Машиностроительный Завод 2023",
            0,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            align="C",
        )
class Globus:
    def __init__(self):
        self.reset()
        self.savedir = ""
    def reset(self):
        self.calc_volume_sum = 0.1
        self.inserted_volume_sum = 0
        self.r = 40  # радиус ВФ
        self.bucket_volume = 1.93
        self.target_radius = 50  # требуемый радиус сваи
        self.mean_column_radius = 0
        self.volume_dict_cyl = {}
        self.meta = {
            "Свая": "",
            "Проект": "",
            "Адрес": "",
            "Подрядчик": "",
            "Заказчик": "",
            "Шкаф управления": "",
            "Буровая": "",
            "Номер": "",
        }
        self.calc = {
            "Количество бакетов": 0,
            "Загруженный объем": 0,
            "Средний диаметр": 0,
            "Глубина сваи": 0,
        }
        self.time = {
            "Дата": 0,
            "Время старта работы": 0,
            "Время завершения работы": 0,
            "Полное время работы": 0,
            "Время бурения": 0,
            "Время уплотнения": 0,
            "Время паузы": 0,
        }
        self.cell_height = 8
        self.filetype = -1
g = Globus()
def set_meta(filename: str):
    """
    Checks 1st line of file, sets g.filetype depending on type of file, then writes metadata into g.meta dictionary
    """
    if filename.split(".")[-1] == "csv":
        Invoker(filename)
        filename=filename[:-3:]+"guh"
    with open(filename, 'rb') as f:
        head_b = f.readline()
        encoding = detect(head_b)
        head = head_b.decode(encoding=encoding['encoding'])
    if filename.split(".")[-1] == "guh" and head.startswith("GHSYS"):
        g.filetype = "guh"
        metadata = head.strip().split(";")
        g.meta.update(
            {
                "Свая": metadata[1],
                "Проект": metadata[2],
                "Адрес": metadata[3],
                "Подрядчик": metadata[4],
                "Заказчик": metadata[5],
                "Шкаф управления": metadata[6],
            }
        )
        try:
            with open('ccc.hdr', 'r') as f:
                for line in f:
                    if line.split(' ')[0] == 'Диаметр:':
                        g.target_radius = int((line.split()[1]))/ 2
                        break
        except Exception:
            pass  # Target radius acquisition error
def create_df_full(filename):
    skips = [0, 2] if g.filetype == "guh" else 1
    df_full = pd.read_csv(filename, sep=";", skiprows=skips, encoding_errors="ignore")
    df_full["Date"] = df_full["Date"].apply(
        lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S")
    )
    df_full["Time"] = df_full["Date"].apply(
        lambda x: str(x - df_full["Date"][0]).split(" ")[-1]
    )
    return df_full
def create_df(df_full):
    df = df_full[df_full.Buckets != 0]
    df = df.drop(
        columns=[
            "Zone",
            "Level",
            "P_Water",
            "P_Air",
            "X_Axis",
            "Y_Axis",
            "Weight",
            "Load",
        ]
    )
    df["New_depth"] = df.Depth[df.Depth.shift(1) != df.Depth]
    df = df[df["New_depth"].notna()]
    df["Down"] = df.New_depth[
        (df.New_depth.shift(1) > df.New_depth) & (df.New_depth.shift(-1) > df.New_depth)
    ]
    df["Up"] = df.New_depth[
        (df.New_depth.shift(1) < df.New_depth) & (df.New_depth.shift(-1) < df.New_depth)
    ]
    df = df[(df["Down"].notna()) | (df["Up"].notna())]
    df["Change"] = df["Down"].notna()
    df["Cm_depth"] = df["Depth"].apply(lambda x: int(abs(x) * 100) if x < 0 else 0)
    return df
def depth_func(df):
    max_depth = df.Cm_depth.max()
    g.volume_dict_cyl = {i: 0 for i in range(max_depth + 1)}
    return max_depth
def graph_builder(df):
    volume = 0  # объем последнего шага
    cycle_start_cm = 0  # нижняя точка цикла
    pressure_start_cm = 0  # верхняя точка цикла
    for i, k in df.T.items():
        if k.Change == 1:  # cycle bottom
            cycle_start_cm = k.Cm_depth
        else:  # cycle top
            if cycle_start_cm == 0:
                continue
            pressure_start_cm = k.Cm_depth
            volume = (cycle_start_cm - pressure_start_cm) * math.pi * g.r**2
            q = pressure_start_cm
            while q < cycle_start_cm:
                v = math.pi * g.r**2
                g.calc_volume_sum += v
                g.volume_dict_cyl[q] += v
                q += 1
            g.inserted_volume_sum += volume
    k_v = (df.Buckets.max() * g.bucket_volume * 1000000) / g.calc_volume_sum
    return k_v
def plotter(df_full, max_depth, k_v):
    column_cyl = pd.Series(g.volume_dict_cyl)
    width_dict_cyl = {}
    for i, k in column_cyl.items():
        width_dict_cyl[i] = math.sqrt((k * k_v) / math.pi)
    rad_cyl = pd.Series(width_dict_cyl)
    g.mean_column_radius = math.sqrt(column_cyl.sum() * k_v / (math.pi * max_depth))
    df_column = pd.DataFrame()
    df_column.insert(0, "Cylinder_rad", rad_cyl)
    df_column["Cylinder_mean"] = df_column["Cylinder_rad"].rolling(16).mean()
    df_column["Cylinder_mean_r"] = df_column["Cylinder_mean"] * (-1)
    df_column = df_column.drop(columns="Cylinder_rad")
    df_column["Depth"] = df_column.index / (-100)
    left_linewidth = 0.5
    right_linewidth = left_linewidth
    font_size = 5
    pos_dif = -0.07
    fig, (ax1, ax0) = plt.subplots(
        ncols=2, gridspec_kw={"width_ratios": [2, 1]}, figsize=(8, 4)
    )
    plt.rcParams["font.size"] = f"{font_size}"
    plt.grid(visible=True, which="minor", alpha=0.5, linewidth=left_linewidth / 2)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(font_size)
    df_full.plot(
        x="Time",
        y="Amperage",
        ax=ax1,
        linewidth=left_linewidth,
        color="tab:red",
        legend=False,
    )
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.tick_params(axis="x", rotation=55)
    ax1.minorticks_on()
    ax1.grid(color="black", alpha=0.5, linewidth=left_linewidth / 2)
    ax1.set_ylim(0, 400)
    ax1.set_ylabel("Сила тока, А", color="tab:red", fontsize=font_size)
    ax1.set_xlabel("Время", fontsize=font_size)
    ax2 = ax1.twinx()
    df_full.plot(
        x="Time",
        y="Depth",
        ax=ax2,
        linewidth=left_linewidth,
        color="tab:green",
        legend=False,
    )
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.minorticks_on()
    ax2.set_ylim(-20, 0)
    ax2.set_ylabel("Глубина, м", color="tab:green", fontsize=font_size)
    ax3 = ax1.twinx()
    df_full.plot(
        x="Time",
        y="Buckets",
        ax=ax3,
        linewidth=left_linewidth,
        color="tab:blue",
        legend=False,
    )
    ax3.tick_params(axis="y", labelcolor="tab:blue")
    ax3.set_ylim(0, 40)
    ax3.set_ylabel("Бакеты, шт", color="tab:blue", fontsize=font_size)
    ax3.spines["left"].set_position(("axes", pos_dif * 2))
    ax3.spines["left"].set_visible(True)
    ax3.yaxis.set_label_position("left")
    ax3.yaxis.set_ticks_position("left")
    ax3.minorticks_on()
    ax0.fill_betweenx(
        y=df_column["Depth"].values,
        x1=df_column["Cylinder_mean"].values,
        x2=df_column["Cylinder_mean_r"].values,
        color="#ccc",
    )
    for label in ax0.get_xticklabels() + ax0.get_yticklabels():
        label.set_fontsize(font_size)
    ax0.plot(
        [-g.target_radius, -g.target_radius],
        [-20, 0],
        color="black",
        linestyle="--",
        linewidth=right_linewidth,
    )
    ax0.plot(
        [g.target_radius, g.target_radius],
        [-20, 0],
        color="black",
        linestyle="--",
        linewidth=right_linewidth,
    )
    ax0.set_ylim(-20, 0)
    ax0.tick_params(axis="y", labelcolor="tab:green")
    ax0.set_xlim(-100, 100)
    ax0.set_xlabel("Радиус, см", fontsize=font_size)
    ax0.minorticks_on()
    ax0.grid(color="black", alpha=0.5, linewidth=right_linewidth / 2)
    fig.tight_layout()
    plt.grid(visible=True, which="minor", alpha=0.5, linewidth=left_linewidth / 2)
    plt.savefig(
        "temp.png",
        orientation="portrait",
        dpi=200,
        bbox_inches="tight",
    )
def set_calc_and_time(df, df_full, max_depth, k_v):
    g.calc.update(
        {
            "Количество бакетов": (df.Buckets.max(), "шт."),
            "Загруженный объем": (
                round(g.inserted_volume_sum * k_v / 1000000, 2),
                "куб. м",
            ),
            "Средний диаметр": (math.floor(g.mean_column_radius * 2), "см"),
            "Глубина сваи": (round(max_depth / 100), "м"),
        }
    )
    g.time.update(
        {
            "Дата": df_full.Date.min().strftime("%d.%m.%Y"),
            "Время старта работы": df_full.Date.min().strftime("%H:%M:%S"),
            "Время завершения работы": df_full.Date.max().strftime("%H:%M:%S"),
            "Полное время работы": df_full.Time.iloc[-1],
            "Время бурения": df.Time.iloc[0],
            "Время уплотнения": str(df.Date.iloc[-1] - df.Date.iloc[0]).split(" ")[-1],
            "Время паузы": 0,
        }
    )
def create_pdf(filename):
    pdf = PDF()
    pdf.add_font("NotoSans", style="", fname="fonts/NotoSans-Regular.ttf")
    pdf.add_font("NotoSans", style="B", fname="fonts/NotoSans-Bold.ttf")
    pdf.add_font("NotoSans", style="I", fname="fonts/NotoSans-Italic.ttf")
    pdf.add_font("NotoSans", style="BI", fname="fonts/NotoSans-BoldItalic.ttf")
    pdf.add_page()
    pdf.set_font("NotoSans", "B", 18)
    pdf.cell(
        w=0,
        h=20,
        txt=f"Паспорт виброуплотненной сваи №{g.meta['Свая']}",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        border=True,
        align="C",
    )
    wl = 53
    wc = 67
    wr = 0
    pdf.set_font("NotoSans", "", 10)
    # for d in g.meta, g.calc, g.time:
    meta_list = []
    calc_list = []
    time_list = []
    for k, v in g.meta.items():
        if v:
            meta_list.append(f"{k.capitalize()}: {v}")
    for k, v in g.calc.items():
        if v:
            try:
                calc_list.append(f"{k.capitalize()}: {v[0]} {v[1]}")
            except Exception:
                calc_list.append(f"{k.capitalize()}: {v}")
    for k, v in g.time.items():
        if v:
            time_list.append(f"{k.capitalize()}: {v}")
    maxlen = max(len(meta_list), len(calc_list), len(time_list))
    for i in range(maxlen):
        if i == 0:
            border = ["LTR", "TR"]
        elif i + 1 == maxlen:
            border = ["LBR", "BR"]
        else:
            border = ["LR", "R"]
        try:
            txt1 = meta_list[i]
        except Exception:
            txt1 = ""
        try:
            txt2 = calc_list[i]
        except Exception:
            txt2 = ""
        try:
            txt3 = time_list[i]
        except Exception:
            txt3 = ""
        pdf.cell(
            w=wl,
            h=g.cell_height,
            txt=txt1,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            border=border[0],
        )
        pdf.cell(
            w=wc,
            h=g.cell_height,
            txt=txt2,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            border=border[1],
        )
        pdf.cell(
            w=wr,
            h=g.cell_height,
            txt=txt3,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            border=border[1],
        )
    pdf.cell(w=0, h=g.cell_height, txt="", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image("temp.png", x="C", y=None, w=190, h=0, link="")
    data=CRISTAL_MAIDEN(filename)
    pdf.set_font("NotoSans", "B", size=4)
    spacing=1
    col_width = pdf.w / 9.5 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    row_height = pdf.font_size+1
    for row in data:
        for item in row:
            pdf.cell(col_width, row_height * spacing,
                    txt=item, border=1)
        pdf.ln(row_height * spacing)
    result_filename = (
        f"results/{filename.split('/')[-1].rsplit('.', maxsplit=1)[0]}.pdf"
        if g.savedir == ""
        else f"{g.savedir}/{filename.split('/')[-1].rsplit('.', maxsplit=1)[0]}.pdf"
    )
    try:
        pdf.output(result_filename)
    except Exception:
        Path(f"results").mkdir(parents=True, exist_ok=True)
        pdf.output(result_filename)
def main(filename):
    set_meta(filename)
    if not g.filetype:
        g.reset()
        return
    if filename.split(".")[-1] == "csv":
        filename = filename[:-3:] + "guh"
    df_full = create_df_full(filename)
    df = create_df(df_full)
    max_depth = depth_func(df)
    k_v = graph_builder(df)
    plotter(df_full, max_depth, k_v)
    set_calc_and_time(df, df_full, max_depth, k_v)
    create_pdf(filename)
    g.reset()
layout = [
    [
        sg.LBox([], size=(60, 10), key="-FILESLB-"),
        sg.Input(visible=False, enable_events=True, key="-IN-"),
        sg.FilesBrowse(
            button_text="Выбор файлов",
            file_types=(
                ("Новый формат отчета", "*.csv"),
                ("Старый формат отчета", "*.guh"),
            ),
        ),
    ],
    [
        sg.T(
            "Сохранить отчеты в:",
            tooltip=f"""Если папка не выбрана, отчеты будут сохранены в папку "{p}/results" """,
        ),
        sg.Input(
            visible=True,
            enable_events=True,
            key="-IN1-",
            tooltip=f"""Если папка не выбрана, отчеты будут сохранены в папку "{p}/results" """,
        ),
        sg.FolderBrowse(
            button_text="Выбор папки",
            tooltip=f"""Если папка не выбрана, отчеты будут сохранены в папку "{p}/results" """,
        ),
    ],
    [
        sg.ProgressBar(
            1, orientation="h", size=(20, 20), key="progress", visible=False
        ),
        sg.T(""),
    ],
    [sg.Button("Сформировать отчеты", key="Go"), sg.Button("Выйти", key="Exit")],
]
window = sg.Window("Graph Builder", layout, resizable=True)
progress_bar = window["progress"]
files_list = []
if __name__ == "__main__":
    while True:  # Event Loop
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        if event == "-IN-":
            for file in values["-IN-"].split(";"):
                files_list.append(file)
            window["-FILESLB-"].Update(files_list)
        if event == "-IN1-":
            g.savedir = values["-IN1-"]
        Errors=[]
        Errors_count=0
        if event == "Go":
            i = 0
            length = len(files_list)
            progress_bar.update(i, length, bar_color=("green", "white"), visible=True)
            for file in files_list:
                i += 1
                progress_bar.update(i, length)
                try:
                    main(file)
                except (ValueError, IndexError, KeyError):
                    name=str(file.split('/')[-1])
                    Errors.append(name)
                    Errors_count+=1
            progress_bar.update(visible=False)
            if Errors_count == 0:
                layout1 = [
                    [sg.Text('Все файлы успешно обработаны', size=(35, 1),key='-text-', font='Helvetica 16')]]
                window1 = sg.Window('Диспетчер', layout1, size=(400,50))
                window1.read()
            else:
                layout2 = [
                    [sg.Text("Список ошибок:")],
                    [sg.Listbox(Errors, size=(60, 10))],
                    [sg.Button("Закрыть")]
                ]
                # Создание окна
                window2 = sg.Window("Список ошибок", layout2)
                # Обработка событий
                while True:
                    event, values = window2.read()
                    if event == sg.WINDOW_CLOSED or event == "Закрыть":
                        break
                # Закрытие окна после завершения
                window2.close()
    window.close()
