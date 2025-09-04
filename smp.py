import pygame
import pygame.gfxdraw
import numpy as np
import threading
from eznn import Dense, Sequential
import time

# --- параметры ---
W, H = 800, 600
PLOT_W = 200
N_CLASSES = 4
POINT_RADIUS = 8
CHOICE_BOX_SIZE = 40
CHOICE_BOX_MARGIN = 10
STEP = 3  # шаг сетки меньше — более точная карта

# --- pygame ---
pygame.init()
screen = pygame.display.set_mode((W+PLOT_W, H))
pygame.display.set_caption("Нейросеть строит границы стран (быстро и точно)")
font = pygame.font.SysFont("Arial", 18)
clock = pygame.time.Clock()

# --- данные ---
X_data = []
y_data = []
loss_history = []
current_class = 0

# --- цвета ---
DARK_COLORS = [(80,30,30),(30,80,50),(30,50,80),(80,80,30)]
COLORS = [(255,100,100),(100,255,180),(100,180,255),(255,255,100)]
DARK_BG = (20,20,30)

# --- модель (больше слоёв + tanh для плавности) ---
model = Sequential([
    Dense(2, 128, "tanh"),
    Dense(128, 128, "tanh"),
    Dense(128, 128, "relu"),
    Dense(128, 64, "relu"),
    Dense(64, N_CLASSES, "softmax")
])

lock = threading.Lock()
training = True

# --- асинхронная функция обучения ---
def async_train():
    while training:
        lock.acquire()
        if len(X_data) >= 1:
            X = np.array(X_data)
            y = np.eye(N_CLASSES)[y_data]
            # делаем несколько шагов обучения за кадр для быстрого подстраивания
            for _ in range(5):
                out = model.forward(X)
                loss = -np.mean(np.sum(y * np.log(out + 1e-9), axis=1))
                grad = (out - y)/X.shape[0]
                model.backward(grad, 0.15)  # чуть меньше lr, но больше итераций
            loss_history.append(loss)
        lock.release()
        time.sleep(0.005)  # минимальная пауза

thread = threading.Thread(target=async_train)
thread.start()

# --- функции отрисовки ---
def draw_background():
    bg_surf = pygame.Surface((W,H))
    lock.acquire()
    if len(X_data) >= 1:
        xs = np.linspace(-1,1,W//STEP)
        ys = np.linspace(-1,1,H//STEP)
        xx, yy = np.meshgrid(xs,ys)
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = np.array([model.predict(grid)])
        preds = preds.reshape(H//STEP,W//STEP)
        for i in range(H//STEP):
            for j in range(W//STEP):
                color = DARK_COLORS[int(preds[i,j])]
                pygame.draw.rect(bg_surf,color,(j*STEP,i*STEP,STEP,STEP))
    else:
        bg_surf.fill(DARK_BG)
    lock.release()
    screen.blit(bg_surf,(0,0))

def draw_points():
    for (x,y),c in zip(X_data, y_data):
        px = int((x+1)/2 * W)
        py = int((y+1)/2 * H)
        col = COLORS[c]
        pygame.gfxdraw.filled_circle(screen, px, py, POINT_RADIUS, col)
        pygame.gfxdraw.aacircle(screen, px, py, POINT_RADIUS, (200,200,200))

def draw_loss_plot():
    screen.fill(DARK_BG,(W,0,PLOT_W,H))
    if len(loss_history)>=2:
        plot_h = H-40
        recent_losses = loss_history[-PLOT_W:]
        max_loss = max(recent_losses)
        min_loss = min(recent_losses)
        rng = max_loss-min_loss if max_loss!=min_loss else 1
        scaled = [H-20 - int((l-min_loss)/rng * plot_h) for l in recent_losses]
        for i in range(1,len(scaled)):
            x1 = W + 10 + i-1
            y1 = scaled[i-1]
            x2 = W + 10 + i
            y2 = scaled[i]
            pygame.draw.line(screen,(50,200,255),(x1,y1),(x2,y2),2)
        # подписи каждые 40 пикселей
        for i, val in enumerate(recent_losses):
            if i%40==0:
                text = font.render(f"{val:.2f}",True,(200,200,200))
                screen.blit(text,(W+i,H-18))
    txt = font.render("Ошибка (loss)",True,(200,200,200))
    screen.blit(txt,(W+30,10))

def draw_choice_boxes():
    for i in range(N_CLASSES):
        x = 10 + i*(CHOICE_BOX_SIZE+CHOICE_BOX_MARGIN)
        y = H - CHOICE_BOX_SIZE -10
        pygame.draw.rect(screen,COLORS[i],(x,y,CHOICE_BOX_SIZE,CHOICE_BOX_SIZE))
        if i==current_class:
            pygame.draw.rect(screen,(255,255,255),(x-2,y-2,CHOICE_BOX_SIZE+4,CHOICE_BOX_SIZE+4),3)
        else:
            pygame.draw.rect(screen,(150,150,150),(x-1,y-1,CHOICE_BOX_SIZE+2,CHOICE_BOX_SIZE+2),2)
        text = font.render(str(i+1),True,(255,255,255))
        screen.blit(text,(x+CHOICE_BOX_SIZE//2-6,y+CHOICE_BOX_SIZE//2-10))

# --- основной цикл ---
running = True
while running:
    for e in pygame.event.get():
        if e.type==pygame.QUIT:
            running=False
        elif e.type==pygame.KEYDOWN:
            if e.key==pygame.K_1: current_class=0
            if e.key==pygame.K_2: current_class=1
            if e.key==pygame.K_3: current_class=2
            if e.key==pygame.K_4: current_class=3
        elif e.type==pygame.MOUSEBUTTONDOWN:
            mx,my = e.pos
            if mx>W: continue
            x = (mx/W)*2-1
            y = (my/H)*2-1
            # быстрый append без долгого lock
            lock.acquire()
            X_data.append([x,y])
            y_data.append(current_class)
            lock.release()

    draw_background()
    draw_points()
    draw_loss_plot()
    draw_choice_boxes()
    pygame.display.flip()
    clock.tick(60)

training=False
thread.join()
pygame.quit()
