# README #

### В чём заключается задача? ###

Без абстракции задача подразумевает прохождение лазерного пучка через систему зеркал, с установленными на них движками (для изменения углов наклона и передвижения), а также для подмножества зеркал направленные на них камеры, с необходимостью итогового прохода через два узких экрана токамака, расположенных на расстоянии порядка метра друг от друга.

Токамак в ходе своей работы имеет специфику «прыгать» в разные стороны, а лазеру необходимо сохранять прохождение через оба экрана для диагностики плазмы, то есть стоит задача в юстировке зеркальной системы (которая также может быть подвержена воздействию внешних сил). Для этого требуется наладить «общение» между конфигурацией зеркал и снимками с камер, чтобы знать какое зеркало куда и на сколько надо сдвинуть/повернуть для достижения рабочего режима.

Решение задачи в программе происходит с точки зрения одной из камер при введённых абстракциях:

* Плоскость системы одного зеркала - лист бумаги
* Область плоскости зеркала для последующего корректного отражения - рамка на листе бумаги
* Пятно от лазерного пучка - нарисованная карандашом эллипсоподобная фигура

Необходимо определить взаимное расположение эллипсоподобной фигуры и контуров листа и рамки, корректность фигуры, вектор направления на камеру. 

### Текущая реализация ###

* Поддерживаются 2 режима - live режим при помощи подключённой камеры и работа на предоставленных изображениях
* При помощи контурного анализа и из соображения возможного соотношения площадей по иерархии контуров находятся контуры листа и рамки
* При помощи контурного анализа находится пятно от пучка лазера
* Используя точку пересечения диагоналей и точку центра масс контура листа или рамки, можно найти вектор направления на камеру 
* При помощи теории линейной алгебры выносится вердикт о принадлежности пятна каждому из контуров

### Что надо исправить/добавить (TO DO) ###

* Расширение возможности определения контура листа при высокой освещённости
* Доопределение контура листа и рамки в случае нахождения 4-го угла и части прилежащих к нему сторон за кадром  
* Корректный поиск пятна от пучка в случае его нахождения на линии контура, вне листа
* Расширение функциональности (определение угла обзора камеры, расстояния от камеры, на сколько и куда нужно скорректировать движение пучка лазера) 

### Примеры ###
В папке "Data_jpg" находятся изображения примеры. 
Результаты работы программы лежат в папке "results".
