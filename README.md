Использование:
```
python train.py --name {experiment_name} --data {path to data} --epochs 15 --anneals 10

python create_submission.py --name {sub name} --weights {path to trained model state dict}
```

Путь в ```--data``` должен вести к корню датасета, где лежат папки ```train``` и ```test```

Использовался модифицированный ```Resnet50 + PANet```. ```ReLU``` был заменён на ```Mish```, ```pooling``` слои были заменены на ```maxpooling``` слои, которые также возвращали координаты нейрона с максимальным сигналом (попытка уменьшить потерю пространственного разрешения).
К каждому примеру применялся случайный набор аугументаций. При тренировке использовался ```smooth_l1_loss``` с порогом в 10 пикселей.
Для оптимизации использовался ```SGD with momentum``` с косинусным отжигом.

![image](/pic.png)
