# coding=utf8

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
    """Clase Vector

    Un Vector es una secuencia finita (sistema) de números. Los Vectores se
    pueden construir con una lista o tupla de números. Si el argumento es un
    Vector, el valor devuelto es el mismo Vector. El atributo 'rpr' indica
    al entorno Jupyter el vector debe ser escrito como fila o columna.

    Parámetros:
        sis (list, tuple, Vector) : Sistema de números. Debe ser una lista o
            tupla de números, o bien otro Vector
        rpr (str) : Representación en Jupyter ('columna' por defecto).
            Indica la forma de representar el Vector en Jupyter. Si 
            rpr='fila' se representa en forma de fila. En caso contrario se 
            representa en forma de columna.

    Atributos:
        lista (list): sistema de números almacenado
        n     (int) : número de elementos de la lista
        rpr   (str) : modo de representación en Jupyter

    Ejemplos:
    >>> # Crea un Vector a partir de una lista (o tupla) de números
    >>> Vector( [1,2,3] )   # con lista
    >>> Vector( (1,2,3) )   # con tupla

    Vector([1,2,3])

    >>> # Crea un Vector a partir de otro Vector
    >>> Vector( Vector([1,2,3]) )

    Vector([1,2,3])
    """        

    def __init__(self, sis, rpr='columna'):
        """Inicializa un Vector con una lista, tupla, u otro Vector"""

        if isinstance(sis, (list,tuple)):
            self.lista  =  list(sis)

        elif isinstance(sis, Vector):
            self.lista = sis.lista        

        else:
            raise ValueError('¡el argumento: debe ser una lista, tupla o Vector!')

        self.rpr  =  rpr
        self.n    =  len (self.lista)

    def __or__(self,i):
        """Selector por la derecha

        Extrae la i-ésima componente del Vector, o genera un nuevo vector con 
        las componentes indicadas (los índices comienzan por la posición 1)

        Parámetros:
            i (int, list, tuple): Índice (lista de índices) de los elementos a 
               selecionar

        Resultado:
            número: Cuando i es int, devuelve el componente i-ésimo del Vector.
            Vector: Cuando i es list o tuple, devuelve el Vector formado por los
                componentes indicados en la lista de índices.

        Ejemplos:
        >>> # Seleción de una componente
        >>> Vector([10,20,30]) | 2

        20

        >>> # Creación de un sub-vector a partir de una lista o tupla de índices        
        >>> Vector([10,20,30]) | [2,1,2]
        >>> Vector([10,20,30]) | (2,1,2)

        Vector([20, 10, 20])
        """

        if isinstance(i,int):
            return self.lista[i-1]

        elif isinstance(i, (list,tuple) ):
            return Vector ([ (self|a) for a in i ])
        
    def __ror__(self,i):
        """Selector por la izquierda

        Hace lo mismo que el método __or__ solo que operando por la izquierda
        """    

        return self | i
    def __add__(self, other):
        """Devuelve el Vector resultante de sumar dos Vectores

        Parámetros: 
            other (Vector): Otro vector con el mismo número de elementos

        Ejemplo
        >>> Vector([10, 20, 30]) + Vector([-1, 1, 1])

        Vector([9, 21, 31])        
        """
        
        if isinstance(other, Vector):
            if self.n == other.n:
                return Vector ([ (self|i) + (other|i) for i in range(1,self.n+1) ])

            else:
                print("error en la suma: vectores con distinto número de componentes")
            
    def __rmul__(self, x):
        """Multiplica un Vector por un número o Vector que estén a su izquierda

        Parámetros:
            x (int, float o Fraction): Número por el que se multiplica
              (Vector): Vector con el mismo número de componentes.

        Resultado:
            Vector: Cuando x es int, float o Fraction, devuelve el Vector que 
                resulta de multiplicar cada componente por x
            Número: Cuando x es Vector, devuelve el producto punto entre 
                vectores (producto escalar usual en R^n)

        Ejemplos:
        >>> 3 * Vector([10, 20, 30]) 

        Vector([30, 60, 90])        

        >>> Vector([1, 1, 1]) * Vector([10, 20, 30])

        60
        """

        if isinstance(x, (int, float, Fraction)):
            return Vector ([ x*(self|i) for i in range(1,self.n+1) ])

        elif isinstance(x, Vector): 
            if self.n == x.n:
                return sum([ (x|i)*(self|i) for i in range(1,self.n+1) ])
            else:
                print("error en producto: vectores con distinto número de componentes")
    def __mul__(self, x):
        """Multiplica un Vector por un número o Matrix a su derecha.

        Parámetros:
            x (int, float o Fraction): Número por el que se multiplica
              (Matrix): Matrix con tantas filas como componentes tiene el Vector

        Resultado:
            Vector: Cuando x es int, float o Fraction, devuelve el Vector que
               resulta de multiplicar cada componente por x
                    Cuando x es Matrix, devuelve el Vector combinación lineal de
               las filas de Matrix (componentes del Vector son los coeficientes 
               de la combinación lineal)

        Ejemplos:
        >>> Vector([10, 20, 30]) * 3

        Vector([30, 60, 90])

        >>> a = Vector([1, 1])
        >>> B = Matrix([Vector([1, 2]), Vector([1, 0]), Vector([9, 2])])
        >>> a * B

        Vector([3, 1, 11])
        """

        if isinstance(x, (int, float, Fraction)):
            return x*self

        elif isinstance(x, Matrix):
            if self.n == x.m:
                return Vector( (~x)*self, rpr='fila')
            else:
                print("error en producto: Vector y Matrix incompatibles")

    def __eq__(self, other):
        """Indica si es cierto que dos vectores son iguales"""
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
    """Clase Matrix

    Una Matrix es una secuencia finita (sistema) de Vectores con el mismo 
    número de componentes. Una Matrix se puede construir con una lista o 
    tupla de Vectores con el mismo número de componentes (serán las columnas
    de la matriz); una lista (o tupla) de listas o tuplas con el mismo 
    número de componentes (serán las filas de la matriz); una Matrix (el 
    valor devuelto será la misma Matrix); una BlockMatrix (el valor devuelto
    es la Matrix correspondiente a la matriz obtenida al unir todos los 
    bloques)

    Parámetros:
        sis (list, tuple, Matrix, BlockMarix): Lista (o tupla) de Vectores 
            con el mismo núm. de componentes (columnas de la matriz); o 
            lista (o tupla) de listas o tuplas con el mismo núm. de 
            componentes (filas de la matriz); u otra Matrix; o una 
            BlockMatrix (matriz particionada por bloques).

    Atributos:
        lista (list): sistema de Vectores almacenado
        m     (int) : número de filas de la matriz
        n     (int) : número de columnas de la matriz

    Ejemplos:
    >>> # Crea una Matrix a partir de una lista de Vectores
    >>> a = Vector( [1,2] )
    >>> b = Vector( [1,0] )
    >>> c = Vector( [9,2] )
    >>> Matrix( [a,b,c] )

    Matrix([Vector([1, 2]), Vector([1, 0]), Vector([9, 2])])

    >>> # Crea una Matrix a partir de una lista de listas de números
    >>> A = Matrix( [ [1,1,9], [2,0,2] ] )
    >>> A

    Matrix([Vector([1, 2]), Vector([1, 0]), Vector([9, 2])])

    >>> # Crea una Matrix a partir de otra Matrix
    >>> Matrix( A )

    Matrix([Vector([1, 2]), Vector([1, 0]), Vector([9, 2])])

    >>> # Crea una Matrix a patir de una BlockMatrix
    >>> Matrix( {1}|A|{2} )

    Matrix([Vector([1, 2]), Vector([1, 0]), Vector([9, 2])])
    """

    def __init__(self, sis):
        """Inicializa una Matrix"""

        
        if isinstance(sis, Matrix):
            self.lista  =  sis.lista

        elif isinstance(sis, BlockMatrix):
            self.lista  =  [Vector([ sis.lista[i][j]|k|s  \
                                  for i in range(sis.m) for s in range(1,(sis.lm[i])+1) ])  \
                                  for j in range(sis.n) for k in range(1,(sis.ln[j])+1) ]
                                  
        elif not isinstance(sis, (str, list, tuple)):
            raise ValueError(\
        '¡argumento: list (tuple) de Vectores (lists o tuples);  BlockMatrix; o Matrix!')
                                    
        elif isinstance(sis[0], (list, tuple)):
            if not all ( (type(sis[0])==type(v)) and (len(sis[0])==len(v)) for v in iter(sis) ):
                raise ValueError('no todas son listas o no tienen la misma longitud!')
    
            self.lista  =  [ Vector([ sis[i][j] for i in range(len(sis   )) ]) \
                                                for j in range(len(sis[0])) ]                                                
        
        elif isinstance(sis[0], Vector):
            if not all ( isinstance(v, Vector) and (sis[0].n == v.n) for v in iter(sis)):
                raise ValueError('no todos son vectores, o no tienen la misma longitud!')

            self.lista  =  list(sis)

        self.m  =  self.lista[0].n
        self.n  =  len(self.lista)
    def __or__(self,j):
        """
        Extrae la i-ésima columna de Matrix; o crea una Matrix con las columnas
        indicadas; o crea una BlockMatrix particionando una Matrix por las
        columnas indicadas (los índices comienzan por la posición 1)

        Parámetros:
            j (int, list, tuple): Índice (lista de índices) de las columnas a 
                  selecionar
              (set): Conjunto de índices de las columnas por donde particionar

        Resultado:
            Vector: Cuando j es int, devuelve la columna j-ésima de Matrix.
            Matrix: Cuando j es list o tuple, devuelve la Matrix formada por las
                columnas indicadas en la lísta de índices.
            BlockMatrix: Cuando j es un set, devuelve la BlockMatrix resultante
                de particionar la matriz por las columnas indicadas

        Ejemplos:
        >>> # Extrae la j-ésima columna la matriz 
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | 2

        Vector([0,2])

        >>> # Matrix formada por Vectores columna indicados en la lista (tupla)
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | [2,1]
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | (2,1)

        Matrix( [Vector([0,2]), Vector([1,0])] )

        >>> # BlockMatrix de la partición de la matriz por la segunda columna
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | {2}

        BlockMatrix( [ [ Matrix([Vector([1, 0]), Vector([0, 2])]), 
                         Matrix([Vector([3, 0])]) ] ] )
        """
        if isinstance(j,int):
            return self.lista[j-1]
            
        elif isinstance(j, (list,tuple)):
            return Matrix ([ self|a for a in j ])
            
        elif isinstance(j,set):
            return BlockMatrix ([ [self|a for a in particion(j,self.n)] ]) 
    def __invert__(self):
        """
        Devuelve la traspuesta de una matriz

        Ejemplo:
        >>> ~Matrix([Vector([1]), Vector([2]), Vector([3])])

        Matrix([Vector([1, 2, 3])])
        """

        return Matrix ([ (self|j).lista for j in range(1,self.n+1) ])
        
    def __ror__(self,i):
        """Operador selector por la izquierda

        Extrae la i-ésima fila de Matrix; o crea una Matrix cuyas filas son las 
        indicadas; o crea una BlockMatrix particionando una Matrix por las filas
        indicadas (los índices comienzan por la posición 1)

        Parámetros:
            i (int, list, tuple): Índice (lista de índices) de las filas a 
                 selecionar
              (set): Conjunto de índices de las filas por donde particionar

        Resultado:
            Vector: Cuando i es int, devuelve la fila i-ésima de Matrix.
            Matrix: Cuando i es list o tuple, devuelve la Matrix cuyas filas son
                las indicadas en la lísta de índices.
            BlockMatrix: Cuando i es un set, devuelve la BlockMatrix resultante
                de particionar la matriz por las filas indicadas

        Ejemplos:
        >>> # Extrae la j-ésima columna la matriz 
        >>> 2 | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])])

        Vector([0, 2, 0])

        >>> # Matrix formada por Vectores fila indicados en la lista (tupla)
        >>> [1,1] | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) 
        >>> (1,1) | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])])

        Matrix([Vector([1, 1]), Vector([0, 0]), Vector([3, 3])])

        >>> # BlockMatrix de la partició de la matriz por la primera fila
        >>> {1} | Matrix([Vector([1,0]), Vector([0,2])])

        BlockMatrix( [ [Matrix([Vector([1]),Vector([0])])],
                       [Matrix([Vector([0]),Vector([2])])] ] )
        """

        if isinstance(i,int):
            return Vector ( (~self)|i, rpr='fila' )
            
        elif isinstance(i, (list,tuple)):        
            return Matrix ( [ (a|self).lista  for a in i ] )
            
        elif isinstance(i,set):
            return BlockMatrix ([ [a|self] for a in particion(i,self.m) ])

    def __add__(self, other):
        """Devuelve la Matrix resultante de sumar dos Matrices

        Parámetros: 
            other (Matrix): Otra Matrix con el mismo número de filas y columnas

        Ejemplo:
        >>> A = Matrix( [Vector([1,0]), Vector([0,1])] )
        >>> B = Matrix( [Vector([0,2]), Vector([2,0])] )
        >>> A + B

        Matrix( [Vector([1,2]), Vector([2,1])] )
        """

        if isinstance(other,Matrix) and self.m == other.m and self.n == other.n:
            return Matrix ([ (self|i) + (other|i) for i in range(1,self.n+1) ])
        else:
            print("error en la suma: matrices con distinto orden")
    def __rmul__(self,x):
        """Multiplica una Matrix por un número a su izquierda.

        Parámetros:
            x (int, float o Fraction): Número por el que se multiplica

        Resultado:
            Matrix: Devuelve el múltiplo de la Matrix

        Ejemplo:
        >>> 10 * Matrix([[1,2],[3,4]])

        Matrix([[10,20], [30,40]])
        """

        if isinstance(x, (int, float, Fraction)):
            return Matrix ([ x*(self|i) for i in range(1,self.n+1) ])

    def __mul__(self,x):
        """Multiplica una Matrix por un número, Vector o Matrix a su derecha

        Parámetros:
            x (int, float o Fraction): Número por el que se multiplica
              (Vector): Vector con tantos componentes como columnas tiene Matrix
              (Matrix): con tantas filas como columnas tiene la Matrix

        Resultado:
            Matrix: Si x es int, float o Fraction, devuelve la Matrix que 
               resulta de multiplicar cada columna por x
            Vector: Si x es Vector, devuelve el Vector combinación lineal de las
               columnas de Matrix (los componentes del Vector son los 
               coeficientes de la combinación)
            Matrix: Si x es Matrix, devuelve el producto entre las matrices
               
        Ejemplos:
        >>> # Producto por un número
        >>> Matrix([[1,2],[3,4]]) * 10

        Matrix([[10,20],[30,40]])

        >>> # Producto por un Vector
        >>> Matrix([Vector([1, 3]), Vector([2, 4])]) * Vector([1, 1])

        Vector([3, 7])

        >>> # Producto por otra Matrix
        >>> Matrix([Vector([1, 3]), Vector([2, 4])]) * Matrix([Vector([1,1])]))

        Matrix([Vector([3, 7])])
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
        """Indica si es cierto que dos matrices son iguales"""
        return self.lista == other.lista
    def __and__(self,t):
        """Transforma las columnas de una Matrix

        Atributos:
            t (T): trasformaciones a aplicar sobre las columnas de Matrix

        Ejemplos:
        >>>  A & T({1,3})                # intercambia las columnas 1 y 3
        >>>  A & T((1,5))                # multiplica la columna 1 por 5
        >>>  A & T((1,2,5))              # suma a la columna 1 la 2 por 5
        >>>  A & T([{1,3},(1,5),(1,2,5)])# aplica la secuencia de transformaciones anteriores
        """

        if isinstance(t.t,set):
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
        """Transforma las filas de una Matrix

        Atributos:
            t (T): trasformaciones a aplicar sobre las filas de Matrix

        Ejemplos:
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
    """Clase T

    T es un objeto que denominaremos tranformación elemental. Guarda en su 
    atributo 't' una abreviatura de una transformación elemental o una 
    secuencia de abreviaturas de transformaciones elementales. Con el método
    __and__ actua sobre otra T para crear una T que es composición de 
    transformaciones elementales (la lista de abreviaturas), o actua sobre 
    una Matrix (para transformar sus filas)

    Atributos:
        t (set)  : {índice, índice}. Abreviatura de un intercambio entre los 
                     vectores correspondientes a dichos índices
          (tuple): (índice, número). Abreviatura de transformación Tipo II 
                     que multiplica el vector correspondiente al índice por 
                     el número 
                 : (índice1, índice2, número). Abreviatura de transformación 
                     Tipo I que suma al vector correspondiente al índice1 el 
                     vector correspondiente al índice2 multiplicado por el 
                     número
          (list) : Lista de conjuntos y tuplas. Secuencia de abreviaturas de
                     transformaciones como las anteriores.             

    Ejemplos:
    >>> # Intercambio entre vectores
    >>> T( {1,2} )

    >>> # Trasformación Tipo II (multiplica por 5 el segundo vector)
    >>> T( (2,5) )

    >>> # Trasformación Tipo I (resta al primer vector el tercero)
    >>> T( (1,3,-1) )

    >>> # Secuencia de las tres transformaciones anteriores
    >>> T( [{1,2}, (2,5), (1,3,-1)] )
    """
    def __init__(self, t):
        """Inicializa una transformación elemental"""        
        self.t = t
    def __and__(self, other):
        def CreaLista(t):
            """Si t es una lista, devuelve t; si no devuelve la lista: [t]"""    

            return ( t if isinstance(t, list) else [t] )
        if isinstance(other, T):
            return T(CreaLista(self.t) + CreaLista(other.t))

        if isinstance(other, Matrix):
            return other.__rand__(self)

    def __repr__(self):
        """ Muestra T en su representación python """
        return 'T(' + repr(self.t) + ')'
class BlockMatrix:
    def __init__(self, sis):
        """Inicializa una BlockMatrix con una lista de listas de matrices"""
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
    """Genera el conjunto clave a partir de una secuencia de tamaños
    número
    >>> key([1,2,1])

    {1, 3, 4}
    """
    return set([ sum(L[0:i]) for i in range(1,len(L)+1) ])   


class V0(Vector):
    def __init__(self, n ,rpr = 'columna'):
        """ Inicializa el vector nulo de n componentes"""

        super(self.__class__ ,self).__init__([0 for i in range(n)], rpr)

class M0(Matrix):

    def __init__(self, m, n=None):
        """ Inicializa una matriz nula de orden n """
        if n is None:
            n = m

        super(self.__class__ ,self).__init__([ V0(m) for j in range(n)])    

class I(Matrix):

    def __init__(self, n):
        """ Inicializa la matriz identidad de tamaño n """

        super(self.__class__ ,self).__init__(\
                      [[(i==j)*1 for i in range(n)] for j in range(n)])


class Normal(Matrix):
    def __init__(self, data):
        """Escalona por Gauss obteniendo una matriz cuyos pivotes son unos"""
        def pivote(v,k):
            """
            Devuelve el primer índice mayor que k de de un 
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
     """Devuelve una BlockMatriz con la solución del problema homogéneo"""
     stack=Matrix(BlockMatrix([[A],[I(A.n)]]))
     soluc=Normal(stack)
     col=soluc.rank[A.m-1]
     return {A.m} | soluc | {col}
