# coding=utf8

# Copyright (C) 2019  Marcor Bujosa

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/

from fractions import Fraction

def html(TeX):
    """ Plantilla HTML para insertar comandos LaTeX """
    return "<p style=\"text-align:center;\">$" + TeX + "$</p>"

def latex(a):
     if isinstance(a,float) | isinstance(a,int):
         return str(a)
     else:
         return a.latex()
def _repr_html_(self): 
    return html(self.latex())

def latex_fraction(self):
    if self.denominator == 1:
         return repr(self.numerator)
    else:
         return "\\frac{"+repr(self.numerator)+"}{"+repr(self.denominator)+"}"     

setattr(Fraction, '_repr_html_', _repr_html_)
setattr(Fraction, 'latex',        latex_fraction)

def inverso(x):
    if x==1 or x == -1:
        return x
    else:
        y = 1/Fraction(x)
        if y.denominator == 1:
             return y.numerator
        else:
             return y

class Vector:
    def __init__(self, sis, rpr='columna'):
        """ Inicializa un vector a partir de distintos tipos de datos:
        1) De una lista o tupla
        >>> Vector([1,2,3])

        Vector([1,2,3])
        2) De otro vector (realiza una copia)
        """        
    
        if isinstance(sis, (list,tuple)):
            self.lista  =  list(sis)
        elif isinstance(sis, Vector):
            self.lista = sis.lista        
        else:
            raise ValueError('¡el argumento: debe ser una lista, tupla o Vector')

        self.rpr  =  rpr
        self.n    =  len (self.lista)

    def __or__(self,i):
        """ Extrae la i-esima componente de un vector por la derecha
        >>> Vector([10,20,30]) | 2

        20

        o un sub-vector a partir de una lista o tupla de índices        
        >>> Vector([10,20,30]) | [2,3]
        >>> Vector([10,20,30]) | (2,3)

        Vector([20, 30])
        """
        if isinstance(i,int):
            return self.lista[i-1]
        elif isinstance(i, (list,tuple) ):
            return Vector ([ (self|a) for a in i ])
        
    def __ror__(self,i):
        """ lo mismo que __or__ solo que por la izquierda
        >>> 1 | Vector([10,20,30])

        10

        >>> [2,3] | Vector([10,20,30])
        >>> (2,3) | Vector([10,20,30])

        Vector([20, 30])
        """    
        return self | i

    def __add__(self, other):
        """ Suma de vectores
        >>> Vector([10,20,30]) + Vector([0,1,1])

        Vector([10,21,31])        
        """
        if isinstance(other, Vector):
            if self.n == other.n:
                return Vector ([ (self|i) + (other|i) for i in range(1,self.n+1) ])
            else:
                print("error en la suma: vectores con distinto número de componentes")
            
    def __rmul__(self, x):
        """ Multiplica un vector por un número a su izquierda
        >>> 3 * Vector([10,20,30]) 

        Vector([30,60,90])        
        """
        if isinstance(x, (int, float, Fraction)):
            return Vector ([ x*(self|i) for i in range(1,self.n+1) ])

        elif isinstance(x, Vector): 
            if self.n == x.n:
                return sum([ (x|i)*(self|i) for i in range(1,self.n+1) ])
            else:
                print("error en producto: vectores con distinto número de componentes")

    def __mul__(self, x):
        """ Multiplica un vector por un número a su derecha
        >>> Vector([10,20,30]) * 3

        Vector([30,60,90])        

        o multiplica un vector por otro (producto escalar usual o producto punto) 
        >>> Vector([1, -1])*Vector([1, 1])

        0
        """
        if isinstance(x, (int, float, Fraction)):
            return x*self

        elif isinstance(x, Matrix):
            if self.n == x.m:
                return Vector( (~x)*self, rpr='fila')
            else:
                print("error en producto: Vector y Matrix incompatibles")

    def __eq__(self, other):
        """a==b es True si a.lista es igual que b.lista. False en caso contrario"""
        return self.lista == other.lista
    def __repr__(self):
        """ Muestra el vector en su representación python """
        return 'Vector(' + repr(self.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el entorno jupyter notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX """
        if self.rpr == 'fila':    
            return '\\begin{pmatrix}' + \
                   ',&'.join([latex(self|i)   for i in range(1,self.n+1)]) + \
                   '\\end{pmatrix}' 
        else:
            return '\\begin{pmatrix}' + \
                   '\\\\'.join([latex(self|i) for i in range(1,self.n+1)]) + \
                   '\\end{pmatrix}'
class Matrix:
    def __init__(self, sis):
        """ Inicializa una matriz a partir de distintos tipos de datos:

        1) De una lista de vectores (columnas)
        >>> Matrix([Vector([1,2,3]),Vector([4,5,6])])

        Matrix([Vector([1,2,3]),Vector([4,5,6])])

        2) De una lista de listas de coeficientes (filas)
        >>> Matrix([[1, 4], [2, 5], [3, 6]])

        Matrix([Vector([1,2,3]),Vector([4,5,6])])

        3) De una BlockMatrix (reune todas las matrices)

        4) De otra matriz (realiza una copia)
        """


        
        if isinstance(sis, Matrix):
            self.lista  =  sis.lista

        elif isinstance(sis, BlockMatrix):
            self.lista  =  [Vector([ sis.lista[i][j]|k|s  \
                                  for i in range(sis.m) for s in range(1,(sis.lm[i])+1) ])  \
                                  for j in range(sis.n) for k in range(1,(sis.ln[j])+1) ]
                                  
        elif not isinstance(sis, (str, list, tuple)):
            raise ValueError('¡el argumento debe ser una lista o tupla de vectores una lista (o tupla) de listas o tuplas, una BlockMatrix o una Matrix!')
                                    
        elif isinstance(sis[0], (list, tuple)):
            it = iter(sis)
            the_len = len(next(it))
            if not all(len(l) == the_len for l in it):
                raise ValueError('no todas las listas (filas) tienen la misma longitud!')
    
            self.lista  =  [ Vector([ sis[i][j] for i in range(len(sis   )) ]) \
                                                for j in range(len(sis[0])) ]                                                

        elif isinstance(sis[0], Vector):
            it = iter(sis)
            the_len = len(next(it).lista)
            if not all(len(l.lista) == the_len for l in it):
                raise ValueError('no todos los vectores (columnas) tienen la misma longitud!')

            self.lista  =  list(sis)

        self.m  =  self.lista[0].n
        self.n  =  len(self.lista)

    def __or__(self,j):
        """ Extrae el i-ésimo vector columna de  una matriz 
        >>> Matrix([[1,2,3],[4,5,6]]) | 2

        Vector([2, 5])

        y también una matriz formada por una serie de vectores columna
        >>> Matrix([[1,2,3],[4,5,6]]) | [2,3]

        Matrix([[2, 3], [5, 6]])

        o

        >>> Matrix([[1,2,3],[4,5,6]]) | (2,3)
        Matrix([[2, 3], [5, 6]])

        y también particiona una matriz por columnas
        >>> Matrix([[1,2,3],[4,5,6],[5,6,7]]) | {2}

        BlockMatrix([[Matrix([[1, 2], [4, 5], [5, 6]]), Matrix([[3], [6], [7]])]])
        """
        """ Extrae el i-ésimo vector fila de  una matriz 
        >>> 2 | Matrix([[1,2,3],[4,5,6],[5,6,7]]) 

        Vector([4, 5, 6])

        y también una sub-matriz a partir de una lista o tupla de índices de filas
        >>> [2,3] | Matrix([[1,2,3],[4,5,6],[5,6,7]]) 
        >>> (2,3) | Matrix([[1,2,3],[4,5,6],[5,6,7]]) 

        Matrix([[4, 5, 6], [5, 6, 7]])

        y también particiona una matriz por filas 
        >>> {2} | Matrix([[1,2,3],[4,5,6],[5,6,7]])

        BlockMatrix([[Matrix([[1, 2, 3], [4, 5, 6]])], [Matrix([[5, 6, 7]])]])
        """
        if isinstance(j,int):
            return self.lista[j-1]
            
        elif isinstance(j, (list,tuple)):
            return Matrix ([ self|a for a in j ])
            
        elif isinstance(j,set):
            return BlockMatrix ([ [self|a for a in particion(j,self.n)] ]) 

    def __invert__(self):
        """ Devuelve la matriz traspuesta
        >>> ~Matrix([[1,2,3]])

        Matrix([[1],[2],[3]]) 
        """
        return Matrix ([ (self|j).lista for j in range(1,self.n+1) ])

    def __ror__(self,i):
        """ Extrae el i-ésimo vector columna de  una matriz 
        >>> Matrix([[1,2,3],[4,5,6]]) | 2

        Vector([2, 5])

        y también una matriz formada por una serie de vectores columna
        >>> Matrix([[1,2,3],[4,5,6]]) | [2,3]

        Matrix([[2, 3], [5, 6]])

        o

        >>> Matrix([[1,2,3],[4,5,6]]) | (2,3)
        Matrix([[2, 3], [5, 6]])

        y también particiona una matriz por columnas
        >>> Matrix([[1,2,3],[4,5,6],[5,6,7]]) | {2}

        BlockMatrix([[Matrix([[1, 2], [4, 5], [5, 6]]), Matrix([[3], [6], [7]])]])
        """
        """ Extrae el i-ésimo vector fila de  una matriz 
        >>> 2 | Matrix([[1,2,3],[4,5,6],[5,6,7]]) 

        Vector([4, 5, 6])

        y también una sub-matriz a partir de una lista o tupla de índices de filas
        >>> [2,3] | Matrix([[1,2,3],[4,5,6],[5,6,7]]) 
        >>> (2,3) | Matrix([[1,2,3],[4,5,6],[5,6,7]]) 

        Matrix([[4, 5, 6], [5, 6, 7]])

        y también particiona una matriz por filas 
        >>> {2} | Matrix([[1,2,3],[4,5,6],[5,6,7]])

        BlockMatrix([[Matrix([[1, 2, 3], [4, 5, 6]])], [Matrix([[5, 6, 7]])]])
        """
        if isinstance(i,int):
            return Vector ( (~self)|i , rpr='fila')
            
        elif isinstance(i, (list,tuple)):        
            return Matrix ([ (a|self).lista  for a in i ])
            
        elif isinstance(i,set):
            return BlockMatrix ([ [a|self] for a in particion(i,self.m) ])

    def __add__(self, other):
        """ Suma de matrices
        >>> Matrix([[10,20], [30,40]]) + Matrix([[1,2], [-30,4]])

        Matrix([[11,22], [0,44]])
        """
        if isinstance(other,Matrix) and self.m == other.m and self.n == other.n:
            return Matrix ([ (self|i) + (other|i) for i in range(1,self.n+1) ])
        else:
            print("error en la suma: matrices con distinto orden")
    def __rmul__(self,x):
        """ Multiplica una matriz por un número a su derecha
        >>> Matrix([[1,2],[3,4]]) * 10

        Matrix([[10,20], [30,40]])
        """
        if isinstance(x, (int, float, Fraction)):
            return Matrix ([ x*(self|i) for i in range(1,self.n+1) ])
    def __mul__(self,x):
        """ Multiplica una matriz por un número a su izquierda
        >>> 10 * Matrix([[1,2],[3,4]]) 

        Matrix([[10,20], [30,40]])
        """
        if isinstance(x, (int, float, Fraction)):
            return x*self

        elif isinstance(x, Vector):
            if self.n == x.n:
                return sum( [(x|j)*(self|j) for j in range(1,self.n+1)], V0(self.m) )
            else:
                print("error en producto: vector y matriz incompatibles")

        elif isinstance(x, Matrix):
            if self.n == x.m:
                return Matrix( [ self*(x|j) for j in range(1,x.n+1)])
            else:
                print("error en producto: matrices incompatibles")

    def __eq__(self, other):
        """A==B es True si A.lista es igual que B.lista. False en caso contrario"""
        return self.lista == other.lista
    def __and__(self,t):
        """ Aplica una o una secuencia de transformaciones elementales por columnas: 
        >>>  A & T({1,3})                 # intercambia las columnas 1 y 3
        >>>  A & T((1,5))                 # multiplica la columna 1 por 5
        >>>  A & T((1,2,5))               # suma a la columna 1 la 2 por 5
        >>>  A & T([{1,3},(1,5),(1,2,5)]) # aplica la secuencia de transformaciones
        """
        if isinstance(t.t,set) and len(t.t) == 2:
            self.lista = Matrix( [(self|max(t.t)) if k==min(t.t) else \
                                      (self|min(t.t)) if k==max(t.t) else \
                                      (self|k) for k in range(1,self.n+1)]).lista

        elif isinstance(t.t,tuple) and len(t.t) == 2:
             self.lista = Matrix([ t.t[1]*(self|k) if k==t.t[0] else (self|k) \
                                   for k in range(1,self.n+1)] ).lista
                  
        elif isinstance(t.t,tuple) and len(t.t) == 3:
             self.lista = Matrix([ (self|k) + t.t[2]*(self|t.t[1]) if k==t.t[0] else \
                                   (self|k) for k in range(1,self.n+1)] ).lista
        elif isinstance(t.t,list):
             for k in t.t:          
                 self & T(k)
        return self

    def __rand__(self,t):
        """ Aplica una o una secuencia de transformaciones elementales por filas: 
        >>>    {1,3} & A               # intercambia las filas 1 y 3
        >>>    (1,5) & A               # multiplica la fila 1 por 5
        >>>  (1,2,5) & A               # suma a la fila 1 la 2 por 5
        
        >>>  [(1,2,5),(1,5),{1,3}] & A # aplica la secuencia de transformaciones
        """
        if isinstance(t.t,set) | isinstance(t.t,tuple):
            self.lista = (~(~self & t)).lista
                  
        elif isinstance(t.t,list):
            for k in reversed(t.t):          
                T(k) & self 

        return self

    def __repr__(self):
        """ Muestra una matriz en su representación python """
        return 'Matrix(' + repr(self.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el  entorno jupyter notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX """
        return '\\begin{bmatrix}' + \
                '\\\\'.join(['&'.join([latex(i|self|j) for j in range(1,self.n+1) ]) \
                                                       for i in range(1,self.m+1) ]) + \
               '\\end{bmatrix}' 
class T:
    def __init__(self, t):
        """ Inicializa una transformación elemental """        
        self.t = t

    def __and__(self,t):
        """ Crea una trasformación composición de dos
        >>> T((1,2)) & T({2,4})

        T([(1,2), {2,4}])

        O aplica la transformación sobre una matriz A
        >>> A & T({1,2})    (intercambia las dos primeras columnas de A)
        """        
        def CreaLista(a):
            """Transforma una una tupla en una lista que contiene la tupla"""
            return (a if isinstance(a,list) else [a])

        if isinstance(t,T):
            return T(CreaLista(self.t) + CreaLista(t.t))

        if isinstance(t,Matrix):
            return t.__rand__(self)

class BlockMatrix:
    def __init__(self, sis):
        """ Inicializa una matriz por bloques usando una lista de listas de matrices.
        """        
        self.lista = list(sis)
        self.m     = len(sis)
        self.n     = len(sis[0])
        self.lm    = [fila[0].m for fila in sis] 
        self.ln    = [c.n for c in sis[0]]

    def __or__(self,j):
        """ Reparticiona por columna una matriz por cajas """
        if isinstance(j,set):
            if self.n == 1:
                return BlockMatrix([ [ self.lista[i][0]|a  \
                                        for a in particion(j,self.lista[0][0].n)] \
                                        for i in range(self.m) ])
                                        
            elif self.n > 1: 
                 return (key(self.lm) | Matrix(self)) | j

        def __ror__(self,i):
            """ Reparticiona por filas una matriz por cajas """
            if isinstance(i,set):
                if self.m == 1:
                    return BlockMatrix([[ a|self.lista[0][j]  \
                                           for j in range(self.n) ] \
                                           for a in particion(i,self.lista[0][0].m)])
                                           
                elif self.m > 1: 
                    return i | (Matrix(self) | key(self.ln))


    def __repr__(self):
        """ Muestra una matriz en su representación python """
        return 'BlockMatrix(' + repr(self.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el  entorno jupyter notebook """
        return html(self.latex())

    def latex(self):
        """ Escribe el código de LaTeX """
        if self.m == self.n == 1:       
            return \
              '\\begin{array}{|c|}' + \
              '\\hline ' + \
              '\\\\ \\hline '.join( \
                    ['\\\\'.join( \
                    ['&'.join( \
                    [latex(self.lista[0][0]) ]) ]) ])  + \
              '\\\\ \\hline ' + \
              '\\end{array}'
        else:
            return \
              '\\left[' + \
              '\\begin{array}{' + '|'.join([n*'c' for n in self.ln])  + '}' + \
              '\\\\ \\hline '.join( \
                    ['\\\\'.join( \
                    ['&'.join( \
                    [latex(self.lista[i][j]|k|s) \
                    for j in range(self.n) for k in range(1,self.ln[j]+1) ]) \
                    for s in range(1,self.lm[i]+1) ]) for i in range(self.m) ])  + \
              '\\\\' + \
              '\\end{array}' + \
              '\\right]'



def particion(s,n):
    """ genera la lista de particionamiento a partir de un conjunto y un número
    >>> particion({1,3,5},7)

    [[1], [2, 3], [4, 5], [6, 7]]
    """
    p = list(s | set([0,n]))
    return [ list(range(p[k]+1,p[k+1]+1)) for k in range(len(p)-1) ]
    
def key(L):
    """ genera el conjunto clave a partir de una secuencia de tamaños
    número
    >>> key([1,2,1])

    {1, 3, 4}
    """
    return set([ sum(L[0:i]) for i in range(1,len(L)+1) ])   


class V0(Vector):
    def __init__(self, n ,rpr = 'columna'):
        """ Inicializa el vector nulo de n componentes"""

        super(self.__class__ ,self).__init__([0 for i in range(n)],rpr)

class M0(Matrix):

    def __init__(self, m, n=None):
        """ Inicializa una matriz nula de orden n """
        if n is None:
            n = m

        super(self.__class__ ,self).__init__( \
                      [[0 for i in range(n)] for j in range(m)])

class I(Matrix):

    def __init__(self, n):
        """ Inicializa la matriz identidad de tamaño n """

        super(self.__class__ ,self).__init__(\
                      [[(i==j)*1 for i in range(n)] for j in range(n)])

class e(Vector):

    def __init__(self, i,n ,rpr = 'columna'):
        """ Inicializa el vector e_i  de tamaño n """

        super(self.__class__ ,self).__init__([((i-1)==k)*1 for k in range(n)],rpr)


class Normal(Matrix):
    def __init__(self, data):
        """ Escalona por Gauss obteniendo una matriz cuyos pivotes son unos """
        def pivote(v,k):
            """ Devuelve el primer índice mayor que k de de un 
            un coeficiente no nulo del vector v. En caso de no existir
            devuelve 0
            """            
            return ([x[0] for x in enumerate(v.lista, 1) \
                                if (x[1] !=0 and x[0] > k)]+[0])[0]

        A = Matrix(data)
        r = 0
        self.rank = []
        for i in range(1,A.n+1):
           p = pivote((i|A),r)
           if p > 0:
              r += 1
              A & T({p,r})
              A & T((r,inverso(i|A|r)))
              A & T([(k, r, -(i|A|k)) for k in range(r+1,A.n+1)])

           self.rank+=[r]
              
        super(self.__class__ ,self).__init__(A.lista)
        
def homogenea(A):
     """ Devuelve una BlockMatriz con la solución del problema homogéneo """
     stack=Matrix(BlockMatrix([[A],[I(A.n)]]))
     soluc=Normal(stack)
     col=soluc.rank[A.m-1]
     return {A.m} | soluc | {col}
