# coding=utf8

from fractions import Fraction

def html(TeX):
    """ Plantilla HTML para insertar comandos LaTeX """
    return "<p style=\"text-align:center;\">$" + TeX + "$</p>"
    
def latex(a):
     if isinstance(a,float) | isinstance(a,int) | isinstance(a,str):
         return str(a)
     else:
         return a.latex()
         
def _repr_html_fraction(self): 
    return html(self.latex())

def latex_fraction(self):
    if self.denominator == 1:
         return repr(self.numerator)
    else:
         return "\\frac{"+repr(self.numerator)+"}{"+repr(self.denominator)+"}"     

setattr(Fraction, '_repr_html_', _repr_html_fraction)
setattr(Fraction, 'latex', latex_fraction)


class Vector:
    """Clase Vector

    Un Vector es una secuencia finita de números. Se puede instanciar con 
    una lista, tupla o Sistema de números. Si se instancia con un Vector se
    crea una copia del mismo. El atributo 'rpr' indica al entorno Jupyter 
    si el vector debe ser escrito como fila o como columna.

    Parámetros:
        sis (list, tuple, Sistema, Vector) : Sistema de números. Debe ser
            una lista, o tupla de números, o bien otro Vector o Sistema.
        rpr (str) : Representación en Jupyter ('columna' por defecto).
            Indica la forma de representar el Vector en Jupyter. Si 
            rpr='fila' se representa en forma de fila. En caso contrario se
            representa en forma de columna.

    Atributos:
        sis   (Sistema): sistema de números almacenado.
        n     (int)    : número de elementos del sistema.
        rpr   (str)    : modo de representación en Jupyter.

    Ejemplos:
    >>> # Instanciación a partir de una lista, tupla o Sistema de números
    >>> Vector( [1,2,3] )           # con lista
    >>> Vector( (1,2,3) )           # con tupla
    >>> Vector( Sistema( [1,2,3] ) )# con Sistema

    Vector([1,2,3])
    >>> # Crear un Vector a partir de otro Vector
    >>> Vector( Vector([1,2,3]) )

    Vector([1,2,3])
    """        
    def __init__(self, data, rpr='columna'):
        """Inicializa Vector con una lista, tupla, Sistema o Vector"""
        if not isinstance(data, (list, tuple, Vector, Sistema) ):
            raise ValueError(' Argumento debe ser una lista, tupla, Sistema o Vector ')
        self.sis  =  Sistema(data)
        self.n    =  len (self.sis)
        self.rpr  =  rpr    

    def __or__(self,i):
        """Selector por la derecha

        Extrae la i-ésima componente o genera un nuevo Vector con las componentes
        indicadas en una lista o tupla (los índices comienzan por la posición 1).

        Parámetros:
            i (int, list, tuple): Índice (o lista de índices) de los elementos
                a seleccionar.
        Resultado:
            número: Cuando i es int, devuelve el componente i-ésimo del Vector.
            Vector: Cuando i es list o tuple, devuelve el Vector formado por los
                componentes indicados en la lista o tupla de índices.
        Ejemplos:
        >>> # Selección de una componente
        >>> Vector([10,20,30]) | 2

        20
        >>> # Creación de un sub-vector a partir de una lista o tupla de índices        
        >>> Vector([10,20,30]) | [2,1,2]
        >>> Vector([10,20,30]) | (2,1,2)

        Vector([20, 10, 20])
        """
        if isinstance(i,int):
            return self.sis|i
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
    
        if not isinstance(other, Vector) or self.n != other.n:
            raise ValueError\
    ('A un Vector solo se le puede sumar otro Vector con el mismo número de componentes')

        return Vector ([ (self|i) + (other|i) for i in range(1,self.n+1) ])
                
    def __sub__(self, other):
        """Devuelve el Vector resultante de restar dos Vectores

        Parámetros: 
            other (Vector): Otro vector con el mismo número de elementos

        Ejemplo
        >>> Vector([10, 20, 30]) - Vector([-1, 1, 1])

        Vector([11, 19, 29])        
        """    
        if not isinstance(other, Vector) or self.n != other.n:
            raise ValueError\
    ('A un Vector solo se le puede restar otro Vector con el mismo número de componentes')

        return Vector ([ (self|i) - (other|i) for i in range(1,self.n+1) ])
                
    def __rmul__(self, x):
        """Multiplica un Vector por un número a su izquierda

        Parámetros:
            x (int, float o Fraction): Número por el que se multiplica

        Resultado:
            Vector: Cuando x es int, float o Fraction, devuelve el Vector que 
                resulta de multiplicar cada componente por x
        Ejemplo:
        >>> 3 * Vector([10, 20, 30]) 

        Vector([30, 60, 90])        
        """
        if isinstance(x, (int, float, Fraction)):
            return Vector ([ x*(self|i) for i in range(1,self.n+1) ])

    def __mul__(self, x):
        """Multiplica un Vector por un número, Matrix o Vector a su derecha.

        Parámetros:
            x (int, float o Fraction): Número por el que se multiplica
              (Vector): Vector con el mismo número de componentes.
              (Matrix): Matrix con tantas filas como componentes tiene el Vector

        Resultado:
            Vector: Cuando x es int, float o Fraction, devuelve el Vector que 
               resulta de multiplicar cada componente por x. 
            Número: Cuando x es Vector, devuelve el producto punto entre vectores
               (producto escalar usual en R^n)
            Vector: Cuando x es Matrix, devuelve la combinación lineal de las
               filas de x (el Vector contiene los coeficientes de la combinación)

        Ejemplos:
        >>> Vector([10, 20, 30]) * 3

        Vector([30, 60, 90])
        >>> Vector([10, 20, 30]) * Vector([1, 1, 1])

        60
        >>> a = Vector([1, 1])
        >>> B = Matrix([Vector([1, 2]), Vector([1, 0]), Vector([9, 2])])
        >>> a * B

        Vector([3, 1, 11])
        """
        if isinstance(x, (int, float, Fraction)):
            return x*self

        elif isinstance(x, Vector): 
            if self.n != x.n:
                raise ValueError('Vectores con distinto número de componentes')
            return sum([ (self|i)*(x|i) for i in range(1,self.n+1) ])

        elif isinstance(x, Matrix):
            if self.n != x.m:
                raise ValueError('Vector y Matrix incompatibles')
            return Vector( (~x)*self, rpr='fila' )

    def __eq__(self, other):
        """Indica si es cierto que dos vectores son iguales"""
        return self.sis == other.sis

    def __ne__(self, other):
        """Indica si es cierto que dos vectores son distintos"""
        return self.sis != other.sis

    def __reversed__(self):
        """Devuelve el reverso de un Vector"""
        return Vector(self.sis.lista[::-1])
    def __neg__(self):
        """Devuelve el opuesto de un Vector"""
        return -1*self

    def esNulo(self):
        """Indica si es cierto que el vector es nulo"""
        return self==self*0

    def __repr__(self):
        """ Muestra el vector en su representación python """
        return 'Vector(' + repr(self.sis.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el entorno jupyter notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para representar un Vector"""
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

    Es un Sistema de Vectores con el mismo número de componentes. Una Matrix
    se puede construir con una lista, tupla o Sistema de: Vectores con el 
    mismo número de componentes (serán las columnas de la matriz); una lista,
    tupla o Sistema de: listas, tuplas o Sistemas con el mismo número de 
    componentes (serán las filas de la matriz); una Matrix (el valor devuelto
    será una copia de la Matrix); una BlockMatrix (el valor devuelto es la 
    Matrix que resulta de unir todos los bloques)

    Parámetros:
        data (list, tuple, Sistema, Matrix, BlockMatrix): Lista, tupla o 
        Sistema de Vectores con el mismo núm. de componentes (columnas); o 
        lista, tupla o Sistema de listas, tuplas o Sistemas con el mismo núm.
        de componentes (filas); u otra Matrix; o una BlockMatrix.

    Atributos:
        sis   (Sistema): Sistema de Vectores (columnas)
        m     (int)    : número de filas de la matriz
        n     (int)    : número de columnas de la matriz

    Ejemplos:
    >>> # Crea una Matrix a partir de una lista de Vectores
    >>> a = Vector( [1,2] )
    >>> b = Vector( [1,0] )
    >>> c = Vector( [9,2] )
    >>> Matrix( [a,b,c] )

    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ])
    >>> # Crea una Matrix a partir de una lista de listas de números
    >>> A = Matrix( [ [1,1,9], [2,0,2] ] )
    >>> A

    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ])
    >>> # Crea una Matrix a partir de otra Matrix
    >>> Matrix( A )

    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ])
    >>> # Crea una Matrix a partir de una BlockMatrix
    >>> Matrix( {1}|A|{2} )

    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ])
    """
    def __init__(self, data):
        """Inicializa una Matrix"""
        lista = Sistema(data).lista
        
        if isinstance(lista[0], Vector):
            if not all ( isinstance(v, Vector) and (lista[0].n == v.n) for v in iter(lista)):
                raise ValueError('no todos son vectores, o no tienen la misma longitud!')

            self.sis  =  Sistema(data)        
                
        elif isinstance(data, BlockMatrix):
            self.sis  =  Sistema([ Vector([ lista[i][j]|k|s  \
                                  for i in range(data.m) for s in range(1,(data.lm[i])+1) ])\
                                  for j in range(data.n) for k in range(1,(data.ln[j])+1) ])
                                                              
        elif isinstance(lista[0], (list, tuple, Sistema)):
            if not all (type(lista[0])==type(v) and len(lista[0])==len(v) for v in iter(lista)):
                raise ValueError('no todas son listas o no tienen la misma longitud!')

            self.sis  =  Sistema([ Vector([ lista[i][j] for i in range(len(lista   )) ]) \
                                                        for j in range(len(lista[0])) ])

        
        self.m  =  self.sis.lista[0].n
        self.n  =  len(self.sis)

    def __or__(self,j):
        """
        Extrae la i-ésima columna de Matrix; o crea una Matrix con las columnas
        indicadas; o crea una BlockMatrix particionando una Matrix por las
        columnas indicadas (los índices comienzan por la posición 1)

        Parámetros:
            j (int, list, tuple): Índice (o lista de índices) de las columnas a 
                  seleccionar
              (set): Conjunto de índices de las columnas por donde particionar

        Resultado:
            Vector: Cuando j es int, devuelve la columna j-ésima de Matrix.
            Matrix: Cuando j es list o tuple, devuelve la Matrix formada por las
                columnas indicadas en la lista o tupla de índices.
            BlockMatrix: Si j es un set, devuelve la BlockMatrix resultante de 
                particionar la matriz por las columnas indicadas en el conjunto

        Ejemplos:
        >>> # Extrae la j-ésima columna la matriz 
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | 2

        Vector([0, 2])
        >>> # Matrix formada por Vectores columna indicados en la lista (o tupla)
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | [2,1]
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | (2,1)

        Matrix( [Vector([0, 2]); Vector([1, 0])] )
        >>> # BlockMatrix correspondiente a la partición por la segunda columna
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | {2}

        BlockMatrix( [ [ Matrix([Vector([1, 0]), Vector([0, 2])]);
                         Matrix([Vector([3, 0])]) ] ] )
        """
        if isinstance(j,int):
            return self.sis|j
        elif isinstance(j, (list,tuple)):
            return Matrix ([ self|a for a in j ])
            
        elif isinstance(j,set):
            return BlockMatrix ([ [self|a for a in particion(j,self.n)] ])
             
    def __invert__(self):
        """
        Devuelve la traspuesta de una matriz.

        Ejemplo:
        >>> ~Matrix([Vector([1]), Vector([2]), Vector([3])])

        Matrix([Vector([1, 2, 3])])
        """
        return Matrix ([ (self|j).sis for j in range(1,self.n+1) ])
        
    def __ror__(self,i):
        """Operador selector por la izquierda

        Extrae la i-ésima fila de Matrix; o crea una Matrix con las filas 
        indicadas; o crea una BlockMatrix particionando una Matrix por las filas
        indicadas (los índices comienzan por la posición 1)

        Parámetros:
            i (int, list, tuple): Índice (o lista de índices) de las filas a 
                 seleccionar
              (set): Conjunto de índices de las filas por donde particionar

        Resultado:
            Vector: Cuando i es int, devuelve la fila i-ésima de Matrix.
            Matrix: Cuando i es list o tuple, devuelve la Matrix cuyas filas son
                las indicadas en la lista de índices.
            BlockMatrix: Cuando i es un set, devuelve la BlockMatrix resultante
                de particionar la matriz por las filas indicadas en el conjunto

        Ejemplos:
        >>> # Extrae la j-ésima fila de la matriz 
        >>> 2 | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])])

        Vector([0, 2, 0])
        >>> # Matrix formada por Vectores fila indicados en la lista (o tupla)
        >>> [1,1] | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) 
        >>> (1,1) | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])])

        Matrix([Vector([1, 1]), Vector([0, 0]), Vector([3, 3])])
        >>> # BlockMatrix correspondiente a la partición por la primera fila
        >>> {1} | Matrix([Vector([1,0]), Vector([0,2])])

        BlockMatrix( [ [Matrix([Vector([1]),Vector([0])])],
                       [Matrix([Vector([0]),Vector([2])])] ] )
        """
        if isinstance(i,int):
            return Vector ( (~self)|i, rpr='fila' )

        elif isinstance(i, (list,tuple)):        
            return Matrix ( [ (a|self).sis  for a in i ] )
            
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

        if not isinstance(other,Matrix) or (self.m,self.n) != (other.m,other.n):
            raise ValueError('A una Matrix solo se le puede sumar otra del mismo orden')
            
        return Matrix ([ (self|i) + (other|i) for i in range(1,self.n+1) ])
            
    def __sub__(self, other):
        """Devuelve la Matrix resultante de restar dos Matrices

        Parámetros: 
            other (Matrix): Otra Matrix con el mismo número de filas y columnas

        Ejemplo:
        >>> A = Matrix( [Vector([1,0]), Vector([0,1])] )
        >>> B = Matrix( [Vector([0,2]), Vector([2,0])] )
        >>> A - B

        Matrix( [Vector([1,-2]), Vector([-2,1])] )
        """
        if not isinstance(other,Matrix) or (self.m,self.n) != (other.m,other.n):
            raise ValueError('A una Matrix solo se le puede restar otra del mismo orden')
            
        return Matrix ([ (self|i) - (other|i) for i in range(1,self.n+1) ])
            
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
        """Multiplica una Matrix por un número, Vector o una Matrix a su derecha

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
            if self.n != x.n:      raise ValueError('Vector y Matrix incompatibles')
            return sum([(x|j)*(self|j) for j in range(1,self.n+1)], V0(self.m))

        elif isinstance(x, Matrix):
            if self.n != x.m:      raise ValueError('matrices incompatibles')
            return Matrix( [ self*(x|j) for j in range(1,x.n+1)] )

    def __eq__(self, other):
        """Indica si es cierto que dos matrices son iguales"""
        return self.sis == other.sis
        
    def __ne__(self, other):
        """Indica si es cierto que dos matrices son distintas"""
        return self.sis != other.sis

    def __and__(self,t):
        """Transforma las columnas de una Matrix

        Atributos:
            t (T): transformaciones a aplicar sobre las columnas de Matrix

        Ejemplos:
        >>>  A & T({1,3})                # Intercambia las columnas 1 y 3
        >>>  A & T((5,1))                # Multiplica la columna 1 por 5
        >>>  A & T((5,2,1))              # suma 5 veces la col. 2 a la col. 1
        >>>  A & T([{1,3},(5,1),(5,2,1)])# Aplica la secuencia de transformac.
                     # sobre las columnas de A y en el mismo orden de la lista
        """
        self.sis = (self.sis & t).copy()
        return self
        
    def __rand__(self,t):
        """Transforma las filas de una Matrix

        Atributos:
            t (T): transformaciones a aplicar sobre las filas de Matrix

        Ejemplos:
        >>>  T({1,3})   & A               # Intercambia las filas 1 y 3
        >>>  T((5,1))   & A               # Multiplica por 5 la fila 1
        >>>  T((5,2,1)) & A               # Suma 5 veces la fila 2 a la fila 1
        >>>  T([(5,2,1),(5,1),{1,3}]) & A # Aplica la secuencia de transformac.
                    # sobre las filas de A y en el orden inverso al de la lista
        """
        if isinstance(t.t,set) | isinstance(t.t,tuple):
            self.sis = (~(~self & t)).sis.copy()
                  
        elif isinstance(t.t,list):
            for k in reversed(t.t):          
                T(k) & self
                
        return self
        
    def __pow__(self,n):
        """Calcula potencias de una Matrix (incluida la inversa)"""
        def MatrixInversa( self ):
            """Calculo de la inversa de una matriz"""
            if self.m != self.n: raise ValueError('Matrix no cuadrada')
            R = ElimrGJ(self)
            if R.rank < R.n:    raise ArithmeticError('Matrix singular')
            return Matrix( I(R.n) & T(R.pasos[1]) )
            
        if self.m != self.n:       raise ValueError('Matrix no cuadrada')
        if not isinstance(n,int):  raise ValueError('La potencia no es un entero')

        M = self
        for i in range(1,abs(n)):
            M = M * self

        return MatrixInversa(M) if n < 0 else M

    def __reversed__(self):
        """Devuelve el reverso de una Matrix"""
        return Matrix(reversed(self.sis))

    def __neg__(self):
        """Devuelve el opuesto de una Matrix"""
        return -1*self

    def K(self,rep=0):
        """Una forma pre-escalonada por columnas (K) de una Matrix"""
        return Elim(self,rep)
        
    def L(self,rep=0): 
        """Una forma escalonada por columnas (L) de una Matrix"""
        return ElimG(self,rep)
        
    def R(self,rep=0):
        """Forma escalonada reducida por columnas (R) de una Matrix"""
        return ElimGJ(self,rep)
        
    def rank(self):
        """Rango de una Matrix"""
        return self.K().rank

    def determinante(self):
        """Devuelve el determinante de una matriz cuadrada"""
        if self.m != self.n:
            raise ValueError('Matrix no es cuadrada')
        A = [ tr for tr in filter( lambda x: len(x)==2, T(self.R().pasos[1]).t ) ]
        m = [-1 if isinstance(tr,set) else Fraction(1,tr[0]) for tr in A]

        producto  = lambda x: 1 if not x else x[0] * producto(x[1:])

        return 0 if self.rank() < self.n else producto(m)

    def GS(self):
        """Devuelve una Matrix equivalente cuyas columnas son ortogonales

        Emplea el método de Gram-Schmidt"""
        A = Matrix(self)
        for n in range(2,A.n+1):
            A & T([ (-Fraction((A|n)*(A|j),(A|j)*(A|j)), j, n) for j in range(1,n) ])
        return A

    def __repr__(self):
        """ Muestra una matriz en su representación python """
        return 'Matrix(' + repr(self.sis) + ')'

    def _repr_html_(self):
        """ Construye la representación para el  entorno jupyter notebook """
        return html(self.latex())
        
    def latex(self):
        """ Construye el comando LaTeX para representar una Matrix """
        return '\\begin{bmatrix}' + \
                '\\\\'.join(['&'.join([latex(i|self|j) for j in range(1,self.n+1) ]) \
                                                       for i in range(1,self.m+1) ]) + \
               '\\end{bmatrix}'
               
    
class T:
    """Clase T

    T ("Transformación elemental") guarda en su atributo 't' una abreviatura
    (o una secuencia de abreviaturas) de transformaciones elementales. Con 
    el método __and__ actúa sobre otra T para crear una T que es composición
    de transformaciones elementales (la lista de abreviaturas), o bien actúa 
    sobre una Matrix (para transformar sus filas)

    Atributos:
        t (set)  : {índice, índice}. Abrev. de un intercambio entre los 
                     vectores correspondientes a dichos índices
          (tuple): (índice, número). Abrev. transf. Tipo II que multiplica
                     el vector correspondiente al índice por el número 
                 : (índice1, índice2, número). Abrev. transformación Tipo I
                     que suma al vector correspondiente al índice1 el vector
                     correspondiente al índice2 multiplicado por el número
          (list) : Lista de conjuntos y tuplas. Secuencia de abrev. de
                     transformaciones como las anteriores.             
          (T)    : Transformación elemental. Genera una T cuyo atributo t es
                     una copia del atributo t de la transformación dada 
          (list) : Lista de transformaciones elementales. Genera una T cuyo 
                     atributo es la concatenanción de todas las abreviaturas
    Ejemplos:
    >>> # Intercambio entre vectores
    >>> T( {1,2} )

    >>> # Trasformación Tipo II (multiplica por 5 el segundo vector)
    >>> T( (5,2) )

    >>> # Trasformación Tipo I (resta el tercer vector al primero)
    >>> T( (-1,3,1) )

    >>> # Secuencia de las tres transformaciones anteriores
    >>> T( [{1,2}, (5,2), (-1,3,1)] )

    >>> # T de una T
    >>> T( T( (5,5) ) )

    T( (5,2) )

    >>> # T de una lista de T's
    >>> T( [T([(-8, 2), (2, 1, 2)]), T([(-8, 3), (3, 1, 3)]) ] )

    T( [(-8, 2), (2, 1, 2), (-8, 3), (3, 1, 3)] )
    """
    def __init__(self, t):
        """Inicializa una transformación elemental"""
        def CreaLista(t):
            """Devuelve t si t es una lista; si no devuelve la lista [t]"""
            return t if isinstance(t, list) else [t]
            
        if isinstance(t, T):
            self.t = t.t

        elif isinstance(t, list) and t and isinstance(t[0], T): 
                self.t = [val for sublist in [x.t for x in t] for val in CreaLista(sublist)]

        else:
            self.t = t
        for j in CreaLista(self.t):
            if isinstance(j,tuple) and (len(j) == 2) and j[0]==0:
                raise ValueError('T( (0, i) ) no es una trasformación elemental')
            if isinstance(j,tuple) and (len(j) == 3) and (j[1] == j[2]):
                raise ValueError('T( (a, i, i) ) no es una trasformación elemental')
            if isinstance(j,set) and (len(j) > 2) or not j:
                raise ValueError \
                ('El conjunto debe tener uno o dos índices para ser trasformación elemental')

    def __and__(self, other):
        """Composición de transformaciones elementales (o transformación filas)

        Crea una T con una lista de abreviaturas de transformaciones elementales
        (o llama al método que modifica las filas de una Matrix)

        Parámetros:
            (T): Crea la abreviatura de la composición de transformaciones, es
                 decir, una lista de abreviaturas
            (Matrix): Llama al método de la clase Matrix que modifica las filas
                 de Matrix
        Ejemplos:
        >>> # Composición de dos Transformaciones elementales
        >>> T( {1, 2} ) & T( (2, 4) )

        T( [{1,2}, (2,4)] )

        >>> # Composición de dos Transformaciones elementales
        >>> T( {1, 2} ) & T( [(2, 4), (2, 1), {3, 1}] )

        T( [{1, 2}, (2, 4), (2, 1), {3, 1}] )

        >>> # Transformación de las filas de una Matrix
        >>> T( [{1,2}, (4,2)] ) & A # multiplica por 4 la segunda fila de A y
                                    # luego intercambia las dos primeras filas
        """        
        def CreaLista(t):
            """Devuelve t si t es una lista; si no devuelve la lista [t]"""
            return t if isinstance(t, list) else [t]
            
        if isinstance(other, T):
            return T(CreaLista(self.t) + CreaLista(other.t))

        if isinstance(other, Matrix):
            return other.__rand__(self)

    def __invert__(self):
        """Transpone la lista de abreviaturas (invierte su orden)"""
        return T( list(reversed(self.t)) ) if isinstance(self.t, list) else self
        
    def __pow__(self,n):
        """Calcula potencias de una T (incluida la inversa)"""
        def Tinversa ( self ):
            """Calculo de la inversa de una transformación elemental"""
            def CreaLista(t):
                """Devuelve t si t es una lista; si no devuelve la lista [t]"""
                return t if isinstance(t, list) else [t]
                

            listaT = [ ( -j[0], j[1], j[2] )    if (isinstance(j,tuple) and len(j)==3) else \
                       (Fraction(1,j[0]), j[1]) if (isinstance(j,tuple) and len(j)==2) else \
                       j                                         for j in CreaLista(self.t) ]

            return ~T( listaT )
    
        if not isinstance(n,int):
            raise ValueError('La potencia no es un entero')
        t = self

        for i in range(1,abs(n)):
            t = t & self

        if n < 0:
            t = Tinversa(t)

        return t

    def espejo ( self ):
        """Calculo de la transformación elemental espejo de otra"""
        def CreaLista(t):
            """Devuelve t si t es una lista; si no devuelve la lista [t]"""
            return t if isinstance(t, list) else [t]
            
        return T([(j[0],j[2],j[1]) if len(j)==3 else j for j in CreaLista(self.t)])
        
    def __repr__(self):
        """ Muestra T en su representación python """
        return 'T(' + repr(self.t) + ')'

    def _repr_html_(self):
        """ Construye la representación para el entorno jupyter notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para representar una Trans. Elem. """
        def simbolo(t):
            """Escribe el símbolo que denota una trasformación elemental particular"""
            if isinstance(t,set):
                return '\\left[\\mathbf{' + latex(min(t)) + \
                  '}\\rightleftharpoons\\mathbf{' + latex(max(t)) + '}\\right]'
            if isinstance(t,tuple) and len(t) == 2:
                return '\\left[\\left(' + \
                  latex(t[0]) + '\\right)\\mathbf{'+ latex(t[1]) + '}\\right]'
            if isinstance(t,tuple) and len(t) == 3:
                return '\\left[\\left(' + latex(t[0]) + '\\right)\\mathbf{' + \
                  latex(t[1]) + '}' + '+\\mathbf{' + latex(t[2]) + '} \\right]'    

        if isinstance(self.t, (set, tuple) ):
            return '\\underset{' + simbolo(self.t) + '}{\\mathbf{\\tau}}'

        elif isinstance(self.t, list):
            return '\\underset{\\begin{subarray}{c} ' + \
                  '\\\\'.join([simbolo(i) for i in self.t])  + \
                  '\\end{subarray}}{\\mathbf{\\tau}}'
                  
class BlockMatrix:
    def __init__(self, data):
        """Inicializa una BlockMatrix con una lista de listas de matrices"""
        self.sis   = Sistema([Sistema(data[i]) for i in range(0,len(data))])
        self.m     = len(data)
        self.n     = len(data[0])
        self.lm    = [fila[0].m for fila in data] 
        self.ln    = [c.n for c in data[0]] 
    def __or__(self,j):
        """ Reparticiona por columna una matriz por cajas """
        if isinstance(j,set):
            if self.n == 1:
                return BlockMatrix([ [ self.sis.lista[i][0]|a  \
                                        for a in particion(j,self.sis.lista[0][0].n)] \
                                        for i in range(self.m) ])
                                        
            elif self.n > 1: 
                 return (key(self.lm) | Matrix(self)) | j

        def __ror__(self,i):
            """ Reparticiona por filas una matriz por cajas """
            if isinstance(i,set):
                if self.m == 1:
                    return BlockMatrix([[ a|self.sis.lista[0][j]  \
                                           for j in range(self.n) ] \
                                           for a in particion(i,self.sis.lista[0][0].m)])
                                           
                elif self.m > 1: 
                    return i | (Matrix(self) | key(self.ln))


    def __repr__(self):
        """ Muestra una matriz en su representación Python """
        return 'BlockMatrix(' + repr(self.sis) + ')'

    def _repr_html_(self):
        """ Construye la representación para el  entorno Jupyter Notebook """
        return html(self.latex())

    def latex(self):
        """ Escribe el código de LaTeX para representar una BlockMatrix """
        if self.m == self.n == 1:       
            return \
              '\\begin{array}{|c|}' + \
              '\\hline ' + \
              '\\\\ \\hline '.join( \
                    ['\\\\'.join( \
                    ['&'.join( \
                    [latex(self.sis.lista[0][0]) ]) ]) ])  + \
              '\\\\ \\hline ' + \
              '\\end{array}'
        else:
            return \
              '\\left[' + \
              '\\begin{array}{' + '|'.join([n*'c' for n in self.ln])  + '}' + \
              '\\\\ \\hline '.join( \
                    ['\\\\'.join( \
                    ['&'.join( \
                    [latex(self.sis.lista[i][j]|k|s) \
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
    p = sorted(list(s | set([0,n])))
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
        n = m if n is None else n

        super(self.__class__ ,self).__init__([ V0(m) for j in range(n)])    

class I(Matrix):
    def __init__(self, n):
        """ Inicializa la matriz identidad de tamaño n """
        super(self.__class__ ,self).__init__(\
                      [[(i==j)*1 for i in range(n)] for j in range(n)])


class Elim(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma pre-escalonada de Matrix(data)

           operando con las columnas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]
            p = ppivote(self, r)
            while p in columnaOcupada:
                p = ppivote(self, p)
            return p
        celim = lambda x: x > p
        A = Matrix(data);  r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T( [ T( [ ( Fraction((i|A|j),(i|A|p)).denominator,   j) ,    \
                               (-Fraction((i|A|j),(i|A|p)).numerator,  p, j)  ] ) \
                                              for j in filter(celim, range(1,A.n+1)) ] )
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(p)
        pasos = [[], transformaciones]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__(A.sis)
        
class ElimG(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada de Matrix(data)

           operando con las columnas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]
            p = ppivote(self, r)
            while p in columnaOcupada:
                p = ppivote(self, p)
            return p
        A = Elim(data);  r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ {p, r} ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(r)
        pasos = [ [], A.pasos[1]+[T(transformaciones)] ]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__(A.sis)

class ElimGJ(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada reducida de Matrix(data)

           operando con las columnas (y evitando operar con fracciones  
           hasta el último momento). Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]
            p = ppivote(self, r)
            while p in columnaOcupada:
                p = ppivote(self, p)
            return p
        celim = lambda x: x < p
        A = ElimG(data);
        r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T( [ T( [ ( Fraction((i|A|j),(i|A|p)).denominator,   j) ,    \
                               (-Fraction((i|A|j),(i|A|p)).numerator,  p, j)  ] ) \
                                              for j in filter(celim, range(1,A.n+1)) ] )
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(p)
                
        transElimIzda = transformaciones

        r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ (Fraction(1, i|A|p), p) ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(p)
                
        pasos = [ [], A.pasos[1] + transElimIzda  + [T(transformaciones)] ]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__(A.sis)

class Elimr(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma pre-escalonada de Matrix(data)

           operando con las columnas. Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]
            p = ppivote(self, r)
            while p in columnaOcupada:
                p = ppivote(self, p)
            return p
        celim = lambda x: x > p
        A = Matrix(data);  r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([(-Fraction(i|A|j, i|A|p), p, j) for j in filter(celim, range(1,A.n+1))])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(p)
        pasos = [[], transformaciones]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__(A.sis)
        
class ElimrG(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada de Matrix(data)

           operando con las columnas. Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]
            p = ppivote(self, r)
            while p in columnaOcupada:
                p = ppivote(self, p)
            return p
        A = Elimr(data);  r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ {p, r} ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(r)
        pasos = [ [], A.pasos[1]+[T(transformaciones)] ]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__(A.sis)

class ElimrGJ(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada reducida de Matrix(data)

           operando con las columnas. Si rep es no nulo, se muestran en
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]
            p = ppivote(self, r)
            while p in columnaOcupada:
                p = ppivote(self, p)
            return p
        celim = lambda x: x < p
        A = ElimrG(data);
        r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([(-Fraction(i|A|j, i|A|p), p, j) for j in filter(celim, range(1,A.n+1))])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(p)                
        transElimIzda = transformaciones
        r = 0;  transformaciones = [];  columnaOcupada = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ (Fraction(1, i|A|p), p) ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                columnaOcupada.add(p)                
        pasos = [ [], A.pasos[1] + transElimIzda  + [T(transformaciones)] ]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__(A.sis)

class ElimF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma pre-escalonada de Matrix(data)

           operando con las filas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        A = Elim(~Matrix(data));     r = A.rank
        pasos = [ list(reversed([ ~t for t in A.pasos[1] ])), [] ]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__((~A).sis)
        
class ElimGF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada de Matrix(data)

           operando con las filas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        A = ElimG(~Matrix(data));    r = A.rank
        pasos = [ list(reversed([ ~t for t in A.pasos[1] ])), [] ]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__((~A).sis)
        
class ElimGJF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada reducida de Matrix(data)

           operando con las columnas (y evitando operar con fracciones  
           hasta el último momento). Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        A = ElimGJ(~Matrix(data));   r = A.rank
        pasos = [ list(reversed([ ~t for t in A.pasos[1] ])), [] ]
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = tex(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        self.rank = r
        super(self.__class__ ,self).__init__((~A).sis)
        
def representa_eliminacion(self, pasos, TexPasosPrev=[], rep=1):
    def tex(data, pasos, TexPasosPrev=[]):
        def PasosYEscritura(data, pasos, TexPasosPrev=[]):
            """Escribe en LaTeX los pasos efectivos dados"""
            A   = Matrix(data);  p   = [[],[]]
            tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
            for l in range(0,2):
                p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                    or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                    or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                            for i in range(0,len(pasos[l])) ]
                p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                if l==0:
                    for i in reversed(range(0,len(p[l]))):
                        tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                        if isinstance (data, Matrix):
                                     tex += latex( p[l][i] & A )
                        elif isinstance (data, BlockMatrix):
                                     tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                if l==1:
                    for i in range(0,len(p[l])):
                        tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                        if isinstance (data, Matrix):
                                     tex += latex( A & p[l][i] )
                        elif isinstance (data, BlockMatrix):
                                     tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
            return tex
        tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(tex))
        return tex
    tex(self, pasos, TexPasosPrev)

class InvMat(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las columnas"""
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        A          = Matrix(data)        
        if A.m != A.n:
            raise ValueError('Matrix no cuadrada')
        M          = ElimGJ(A)
        if M.rank < A.n:
            raise ArithmeticError('Matrix singular')        
        Inv        = I(A.n) & T(M.pasos[1])  
        self.pasos = M.pasos 
        self.tex   = tex( BlockMatrix([ [A], [I(A.n)] ]) , self.pasos)
        super(self.__class__ ,self).__init__(Inv.sis)

class InvMatF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas"""
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        A          = Matrix(data)
        if A.m != A.n:
            raise ValueError('Matrix no cuadrada')
        M          = ElimGJF(A)
        if M.rank < A.n:
            raise ArithmeticError('Matrix singular')        
        Inv        = T(M.pasos[0]) & I(A.n)   
        self.pasos = M.pasos 
        self.tex   = tex( BlockMatrix([ [A,I(A.m)] ]) , self.pasos)
        super(self.__class__ ,self).__init__(Inv.sis)

class InvMatFC(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas y columnas"""
        def PasosYEscritura(data, pasos, TexPasosPrev=[]):
            """Escribe en LaTeX los pasos efectivos dados"""
            A   = Matrix(data);  p   = [[],[]]
            tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
            for l in range(0,2):
                p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                    or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                    or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                            for i in range(0,len(pasos[l])) ]
                p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                if l==0:
                    for i in reversed(range(0,len(p[l]))):
                        tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                        if isinstance (data, Matrix):
                                     tex += latex( p[l][i] & A )
                        elif isinstance (data, BlockMatrix):
                                     tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                if l==1:
                    for i in range(0,len(p[l])):
                        tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                        if isinstance (data, Matrix):
                                     tex += latex( A & p[l][i] )
                        elif isinstance (data, BlockMatrix):
                                     tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
            return tex
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        A          = Matrix(data)
        if A.m != A.n:
            raise ValueError('Matrix no cuadrada')
        M          = ElimGJ(ElimGF(A))
        if M.rank < A.n:
            raise ArithmeticError('Matrix singular')        
        Inv        = ( I(A.n) & T(M.pasos[1]) ) * ( T(M.pasos[0]) & I(A.n) )
        self.pasos = M.pasos  
        self.tex = tex(BlockMatrix([ [A,I(A.m)], [I(A.n),M0(A.m,A.n)] ]),self.pasos)
        super(self.__class__ ,self).__init__(Inv.sis)

class Sistema:
    """Clase Sistema

    Un Sistema es una lista ordenada de objetos. Los Sistemas se instancian
    con una lista, tupla, Vector, Matrix, BlockMatrix, o con otro Sistema 
    de objetos. 

    Parámetros:
        data (list, tuple, Vector, Matrix, BlockMatrix, Sistema): Lista o 
           tupla de objetos (u objeto formado por un Sistema de objetos).

    Atributos:
        sis (list): lista de objetos.

    Ejemplos:
    >>> # Crear un Sistema a partir de una lista (o tupla) de números
    >>> Sistema( [1,2,3] )   # con lista
    >>> Sistema( (1,2,3) )   # con tupla

    [1; 2; 3; 4]

    >>> # Copiar un Sistema o formar un nuevo Sistema copiando el Sistema 
    >>> # de un Vector, Matrix o BlockMatrix
    >>> Sistema( Sistema( [1,2,3] ) )  # copia
    >>> Sistema( A )   # Sistema con los objetos contenidos A.sis (donde A 
                                        es un Vector, Matrix o BlockMatrix)
    """
    def __init__(self, data):
        """Inicializa un Sistema"""
        if isinstance(data, (list, tuple)):
            self.lista = list(data)
        elif isinstance(data, Sistema):
            self.lista = data.lista.copy()
        elif isinstance(data, (Vector, Matrix, BlockMatrix)):
            self.lista = data.sis.lista.copy()
        else:
            raise \
    ValueError('El argumento debe ser una lista, tupla, Sistema, Vector, Matrix\
     o BlockMatrix ')

    def __getitem__(self,i):
        """ Devuelve el i-ésimo coeficiente del Sistema """
        return self.lista[i]

    def __setitem__(self,i,value):
        """ Modifica el i-ésimo coeficiente del Sistema """
        self.lista[i]=value
            
    def __add__(self,other):
        """ Concatena dos Sistemas """
        if not isinstance(other, Sistema):
            raise ValueError('Un Sistema solo se puede concatenar con otro Sistema')
        return Sistema(self.lista + other.lista)
            
    def __len__(self):
        """Número de elementos del Sistema """
        return len(self.lista)

    def copy(self):
        """ Copia la lista de otro Sistema"""
        return Sistema(self.lista.copy())
            
    def __eq__(self, other):
        """Indica si es cierto que dos Sistemas son iguales"""
        return self.lista == other.lista

    def __ne__(self, other):
        """Indica si es cierto que dos Sistemas son distintos"""
        return self.lista != other.lista

    def __reversed__(self):
        """Devuelve el reverso de un Sistema"""
        return Sistema(list(reversed(self.lista)))
        
    def __or__(self,i):
        """
        Extrae el i-ésimo componente del Sistema; o crea un Sistema con los
        elementos indicados (los índices comienzan por la posición 1)

        Parámetros:
            j (int, list, tuple): Índice (o lista de índices) de las columnas a 
                  seleccionar

        Resultado:
                  ?: Cuando j es int, devuelve el elemento j-ésimo del Sistema.
            Sistema: Cuando j es list o tuple, devuelve el Sistema formado por
                  los elementos indicados en la lista o tupla de índices.

        Ejemplos:
        >>> # Extrae el j-ésimo elemento del Sistema 
        >>> Sistema([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | 2

        Vector([0, 2])
        >>> # Sistema formado por los elementos indicados en la lista (o tupla)
        >>> Sistema([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | [2,1]
        >>> Sistema([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | (2,1)

        [Vector([0, 2]); Vector([1, 0])]
        """
        if isinstance(i,int):
            return self.lista[i-1]

        elif isinstance(i, (list,tuple) ):
            return Sistema ([ (self|a) for a in i ])
        
    def __mul__(self,x):
        """Multiplica un Sistema por un  Vector o una Matrix a su derecha

        Parámetros:
            x (Vector): Vector con tantos componentes como elementos tiene el 
                        Sistema
              (Matrix): con tantas filas como elementos tiene el Sistema

        Resultado:
            Combinación de los elementos del Sistema: Si x es Vector, devuelve
               una combinación lineal de los componentes del Sistema, si dicha 
               operación está definida para ellos (los componentes del Vector 
               son los coeficientes de la combinación)
            Matrix: Si x es Matrix, devuelve un Sistema si esa definida la 
               operación combinación lineal entre los objetos del Sistema
               
        Ejemplos:
        >>> # Producto por un Vector
        >>> Sistema([Vector([1, 3]), Vector([2, 4])]) * Vector([1, 1])

        Vector([3, 7])
        >>> # Producto por una Matrix
        >>> Sistema([Vector([1, 3]), Vector([2, 4])]) * Matrix([Vector([1,1])]))

        [Vector([3, 7])]
        """
        if isinstance(x, Vector):
            if len(self) != x.n:
                raise ValueError('Vector y Sistema incompatibles')
            return sum([(x|j)*(self|j) for j in range(1,len(self)+1)][1:], (x|1)*(self|1))

        elif isinstance(x, Matrix):
            if len(self) != x.m:
                raise ValueError('Matrix y Sistema incompatibles')
            return Sistema( [ self*(x|j) for j in range(1,x.n+1)] )
    def __and__(self,t):
        """Transforma los elementos de un Sistema S

        Atributos:
            t (T): transformaciones a aplicar sobre un Sistema S
        Ejemplos:
        >>>  S & T({1,3})                # Intercambia los elementos 1º y 3º
        >>>  S & T((5,1))                # Multiplica por 5 el primer elemento
        >>>  S & T((5,2,1))              # Suma 5 veces el elem. 1º al elem. 2º
        >>>  S & T([{1,3},(5,1),(5,2,1)])# Aplica la secuencia de transformac.
                     # sobre los elementos de S y en el orden de la lista
        """
        if isinstance(t.t,set):
            self.lista = Sistema([(self|max(t.t)) if k==min(t.t) else \
                                  (self|min(t.t)) if k==max(t.t) else \
                                  (self|k) for k in range(1,len(self)+1)] ).lista.copy()

        elif isinstance(t.t,tuple) and (len(t.t) == 2):
            self.lista = Sistema([ t.t[0]*(self|k) if k==t.t[1] else  \
                                  (self|k) for k in range(1,len(self)+1)] ).lista.copy()

        elif isinstance(t.t,tuple) and (len(t.t) == 3):
            self.lista = Sistema([ t.t[0]*(self|t.t[1]) + (self|k) if k==t.t[2] else \
                                  (self|k) for k in range(1,len(self)+1)] ).lista.copy()
        elif isinstance(t.t,list):
            for k in t.t:          
                self & T(k)
        return self
            
    def __repr__(self):
        """ Muestra un Sistema en su representación python """
        return '[' + \
            '; '.join(  repr (self.lista[i]) for i in range(0,len(self.lista)) ) + \
            ']'

    def _repr_html_(self):
        """ Construye la representación para el entorno jupyter notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para representar un Sistema """
        return '\\left[' + \
            ';\;'.join( latex(self.lista[i]) for i in range(0,len(self.lista)) ) + \
            '\\right]'

class SubEspacio:
    def __init__(self,data):
        """Inicializa un SubEspacio de Rn"""
        def SGenENulo(A):
            """Encuentra un sistema generador del Espacio Nulo de A"""
            L = ElimG(A)
            S = Sistema([ (I(A.n)&T(L.pasos[1]))|j for j in range(L.rank+1, A.n+1) ])
            return Sistema([V0(A.n)]) if L.rank==A.n else S
        if not isinstance(data, (Sistema, Matrix)):
            raise ValueError(' Argumento debe ser un Sistema o Matrix ')
        if isinstance(data, Sistema):
            A          = Matrix(data)
            L          = ElimG(A)
            self.dim   = L.rank
            self.base  = Sistema([L|j for j in range(1,L.rank+1)])
            self.sgen  = self.base if L.rank else Sistema([V0(A.m)])
            self.cart  = ~Matrix(SGenENulo(~A))
            self.Rn    = A.m
        if isinstance(data, Matrix):
            A          = data
            self.sgen  = SGenENulo(A)  
            self.dim   = 0 if self.sgen.lista[0].esNulo() else len(self.sgen)
            self.base  = self.sgen if self.dim else Sistema([])
            self.cart  = ~Matrix(SGenENulo(~Matrix(self.sgen)))
            self.Rn    = A.n
    def contenido_en(self, other):
        """Indica si este SubEspacio está contenido en other"""
        self.verificacion(other)
        if isinstance(other, SubEspacio):
            return all ([ (other.cart*v).esNulo() for v in self.sgen ])
        elif isinstance(other, EAfin):
            return other.v.esNulo() and self.contenido_en(other.S)
        else:
            raise ValueError('other debe ser un SubEspacio o un EAfin')

    def __eq__(self, other):
        """Indica si un subespacio de Rn es igual a otro"""
        self.verificacion(other)
        return self.contenido_en(other) and other.contenido_en(self)

    def __ne__(self, other):
        """Indica si un subespacio de Rn es distinto de otro"""
        self.verificacion(other)
        return not (self == other)

    def verificacion(self,other):
        if not isinstance(other, (SubEspacio, EAfin)) or  not self.Rn == other.Rn: 
            raise \
             ValueError('Ambos argumentos deben ser subconjuntos de en un mismo espacio')

    def __add__(self, other):
        """Devuelve la suma de subespacios de Rn"""
        self.verificacion(other)
        return SubEspacio(Sistema(self.sgen + other.sgen))

    def __and__(self, other):
        """Devuelve la intersección de subespacios"""
        self.verificacion(other)
        M = Matrix(BlockMatrix([ [ self.cart], [other.cart] ]))
        return SubEspacio(M)

    def __invert__(self):
        """Devuelve el complemento ortogonal"""
        return SubEspacio(Sistema((~self.cart).sis))

    def __contains__(self, other):
        """Indica si un Vector está pertenece a un SubEspacio"""
        if not isinstance(other, Vector) or other.n != self.cart.n:
            raise ValueError\
                  ('Es necesario un Vector con el número adecuado de componentes')
        return (self.cart*other == V0(self.cart.m))

    def _repr_html_(self):
        """Construye la representación para el entorno jupyter notebook"""
        return html(self.latex())

    def EcParametricas(self):
        """Representación paramétrica del SubEspacio"""
        return '\\left\\{ \\boldsymbol{v}\\in\\mathbb{R}^' \
          + latex(self.Rn) \
          + '\ \\left|\ \\exists\\boldsymbol{p}\\in\\mathbb{R}^' \
          + latex(max(self.dim,1)) \
          + '\ \\text{tal que}\ \\boldsymbol{v}= '\
          + latex(Matrix(self.sgen.lista)) \
          + '\\boldsymbol{p}\\right. \\right\\}' \
          #+ '\qquad\\text{(ecuaciones paramétricas)}'

    def EcCartesianas(self):
        """Representación cartesiana del SubEspacio"""
        return '\\left\\{ \\boldsymbol{v}\\in\\mathbb{R}^' \
          + latex(self.Rn) \
          + '\ \\left|\ ' \
          + latex(self.cart) \
          + '\\boldsymbol{v}=\\boldsymbol{0}\\right.\\right\\}' \
          #+ '\qquad\\text{(ecuaciones cartesianas)}'
        
    def latex(self):
        """ Construye el comando LaTeX para un SubEspacio de Rn"""
        return self.EcParametricas() + '\; = \;' + self.EcCartesianas()
            

class EAfin:
    def __init__(self,data,v):
        """Inicializa un Espacio Afín de Rn"""
        self.S  = data if isinstance(data, SubEspacio) else SubEspacio(data)
        if not isinstance(v, Vector) or v.n != self.S.Rn:
             raise ValueError('v y SubEspacio deben estar en el mismo espacio vectorial')
        MA      = Matrix( BlockMatrix([ [ Matrix(self.S.sgen), Matrix([v]) ] ]) )
        self.v  = Elimr( MA )|0
        self.Rn = self.S.Rn
        
    def __contains__(self, other):
        """Indica si un Vector pertenece a un EAfin"""
        if not isinstance(other, Vector) or other.n != self.S.cart.n:
            raise ValueError('Vector con un número inadecuado de componentes')
        return (self.S.cart)*other == (self.S.cart)*self.v

    def contenido_en(self, other):
        """Indica si este EAfin está contenido en other"""
        self.verificacion(other)
        if isinstance(other, SubEspacio):
             return self.v in other and self.S.contenido_en(other)
        elif isinstance(other, EAfin):
             return self.v in other and self.S.contenido_en(other.S)
        else:
             raise ValueError('other debe ser un SubEspacio o un EAfin')

    def __eq__(self, other):
        """Indica si un EAfin de Rn es igual a other"""
        self.verificacion(other)
        return self.contenido_en(other) and other.contenido_en(self)

    def __ne__(self, other):
        """Indica si un subespacio de Rn es distinto de other"""
        self.verificacion(other)
        return not (self == other)

    def verificacion(self,other):
        if not isinstance(other, (SubEspacio, EAfin)) or  not self.Rn == other.Rn: 
            raise \
             ValueError('Ambos argumentos deben ser subconjuntos de en un mismo espacio')

    def __and__(self, other):
        """Devuelve la intersección de este EAfin con other"""
        self.verificacion(other)
        if isinstance(other, EAfin):
            M = Matrix(BlockMatrix([ [ self.S.cart], [other.S.cart] ]))
            w = Vector((self.S.cart*self.v).sis+(other.S.cart*other.v).sis)
        elif isinstance(other, SubEspacio):
            M = Matrix(BlockMatrix([ [ self.S.cart], [other.cart] ]))
            w = Vector((self.S.cart*self.v).sis+(other.S.cart*V0(S.Rn)).sis)
        try:
            S=SEL(M,w)
        except:
            print('Intersección vacía')
            return Sistema([])
        else:
            return S.eafin

    def __invert__(self):
        """Devuelve el mayor SubEspacio perpendicular a self"""
        return SubEspacio(Sistema((~self.S.cart).sis))

    def _repr_html_(self):
        """Construye la representación para el entorno jupyter notebook"""
        return html(self.latex())

    def EcParametricas(self):
        """Representación paramétrica del SubEspacio"""
        return '\\left\\{ \\boldsymbol{v}\\in\\mathbb{R}^' \
          + latex(self.S.Rn) \
          + '\ \\left|\ \\exists\\boldsymbol{p}\\in\\mathbb{R}^' \
          + latex(max(self.S.dim,1)) \
          + '\ \\text{tal que}\ \\boldsymbol{v}= '\
          + latex(self.v) + '+' \
          + latex(Matrix(self.S.sgen.lista)) \
          + '\\boldsymbol{p}\\right. \\right\\}' \
          #+ '\qquad\\text{(ecuaciones paramétricas)}'

    def EcCartesianas(self):
        """Representación cartesiana del SubEspacio"""
        return '\\left\\{ \\boldsymbol{v}\\in\\mathbb{R}^' \
          + latex(self.S.Rn) \
          + '\ \\left|\ ' \
          + latex(self.S.cart) \
          + '\\boldsymbol{v}=' \
          + latex(self.S.cart*self.v) \
          + '\\right.\\right\\}' \
          #+ '\qquad\\text{(ecuaciones cartesianas)}'
        
    def latex(self):
        """ Construye el comando LaTeX para un SubEspacio de Rn"""
        if self.v != 0*self.v:
             return self.EcParametricas() + '\; = \;' + self.EcCartesianas()
        else:
             return latex(self.S)
            
class Homogenea:
    def __init__(self, data, rep=0):
        """Resuelve un Sistema de Ecuaciones Lineales Homogeneo
    
        y muestra los pasos para encontrarlo"""
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        
        A     = Matrix(data)
        L     = Elim( A )  
        E     = I(A.n) & T(L.pasos[1])
        base  = [Vector(E|j) for j in range(1, L.n+1) if Vector(L|j).esNulo()]
        dim   = len(base)
        
        self.sgen        = Sistema(base) if dim else Sistema([V0(A.n)])
        self.determinado = (dim == 0)
        self.pasos       = L.pasos
        self.tex         = tex( BlockMatrix([[A],[I(A.n)]]), self.pasos)
        self.enulo       = SubEspacio(self.sgen)
        
    def __repr__(self):
        """Muestra el Espacio Nulo de una matriz en su representación python"""
        return 'Combinaciones lineales de (' + repr(self.sgen) + ')'

    def _repr_html_(self):
        """Construye la representación para el entorno jupyter notebook"""
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para la solución de un Sistema Homogéneo"""
        if self.determinado:
            return '\\text{La única solución es el vector cero: }' + \
                         latex(self.sgen.lista[0]) 
        else:
            return '\\text{Conjunto de combinaciones lineales de }' + \
             ',\;'.join([latex(self.sgen.lista[i]) for i in range(0,len(self.sgen))]) 
   
           
class SEL:
    def __init__(self, A, b, rep=0):
        """Resuelve un Sistema de Ecuaciones Lineales

        mediante eliminación por columas en la matriz ampliada y muestra
        los pasos dados"""
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex        
        A  = Matrix(A)
        MA = A if b.esNulo() else Matrix( BlockMatrix([ [A, Matrix([-b])] ]) )
        BM = Matrix( BlockMatrix([ [MA], [I(MA.n)] ]) )
        filasA = list(range(1,A.m+1))
        filasE = columnasE = list(range(1,A.n+1))
        
        LA = Elim ( MA )
        
        if not b.esNulo() and not (filasA|LA|0).esNulo():
            self.tex = tex( {A.m, A.m+A.n} | BM | {A.n}, LA.pasos )
            raise ArithmeticError('No hay solución: Sistema incompatible')
        EA        = I(MA.n) & T(LA.pasos[1])
        Normaliza = T([])    if b.esNulo() else T([(Fraction(1,0|EA|0),MA.n)])
        EA        = EA & Normaliza
        self.solP = V0(MA.n) if b.esNulo() else filasE|EA|0

        E                = filasE| EA |columnasE
        base             = [ (E|j) for j in columnasE if (filasA|LA|j).esNulo()]
        self.determinado = (len(base) == 0)
        self.sgen        = Sistema([V0(A.n)]) if self.determinado else Sistema(base)
        self.eafin       = EAfin(self.sgen,self.solP)

        self.pasos       = [ [], LA.pasos[1] + [Normaliza] ]
        self.tex         = tex( {A.m, A.m+A.n} | BM | {A.n}, self.pasos )
    def EcParametricas(self):
        """Representación paramétrica del SubEspacio"""
        return '\\left\\{ \\boldsymbol{x}\\in\\mathbb{R}^' \
          + latex(self.eafin.Rn) \
          + '\ \\left|\ \\exists\\boldsymbol{p}\\in\\mathbb{R}^' \
          + latex(len(self.sgen)) \
          + '\ \\text{tal que}\ \\boldsymbol{x}= '\
          + latex(self.solP) + '+' \
          + latex(Matrix(self.sgen)) \
          + '\\boldsymbol{p}\\right. \\right\\}' \
       
    def __repr__(self):
        """Muestra el Espacio Nulo de una matriz en su representación python"""
        return repr(self.solP) + ' + Combinaciones lineales de (' + repr(self.sgen) + ')'

    def _repr_html_(self):
        """Construye la representación para el entorno jupyter notebook"""
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para la solución de un Sistema Homogéneo"""
        if self.determinado:
            return '\\text{Tiene solución única:  }\\boldsymbol{x}=' + latex(self.solP) 
        else:
            return '\\text{Conjunto de vectores: }' + self.EcParametricas()
              
class SELS:
    def __init__(self, A, b, rep=0):
        """Resuelve un Sistema de Ecuaciones Lineales
    
        mediante eliminación por columas en la matriz por bloques que
        incluye la matriz identidad y muestra los pasos dados"""
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        A  = Matrix(A)      
        MA = A if b.esNulo() else Matrix( BlockMatrix([ [A, Matrix([-b])] ]) )
        BM = Matrix( BlockMatrix([ [MA], [I(MA.n)] ]) )
        filasA = list(range(1,A.m+1))
        filasE = columnasE = list(range(1,A.n+1))
        
        LA = Elim ( BM )
        
        if not b.esNulo() and not (filasA|LA|0).esNulo():
            self.tex = tex( {A.m, A.m+A.n} | BM | {A.n}, LA.pasos )
            raise ArithmeticError('No hay solución: Sistema incompatible')
        EA        = I(MA.n) & T(LA.pasos[1])
        Normaliza = T([])    if b.esNulo() else T([(Fraction(1,0|EA|0),MA.n)])
        EA        = EA & Normaliza
        self.solP = V0(MA.n) if b.esNulo() else filasE|EA|0

        E                = filasE| EA |columnasE
        base             = [ (E|j) for j in columnasE if (filasA|LA|j).esNulo()]
        self.determinado = (len(base) == 0)
        self.sgen        = Sistema([V0(A.n)]) if self.determinado else Sistema(base)
        self.eafin       = EAfin(self.sgen,self.solP)

        self.pasos       = [ [], LA.pasos[1] + [Normaliza] ]
        self.tex         = tex( {A.m, A.m+A.n} | BM | {A.n}, self.pasos )    
    def EcParametricas(self):
        """Representación paramétrica del SubEspacio"""
        return '\\left\\{ \\boldsymbol{x}\\in\\mathbb{R}^' \
          + latex(self.eafin.Rn) \
          + '\ \\left|\ \\exists\\boldsymbol{p}\\in\\mathbb{R}^' \
          + latex(len(self.sgen)) \
          + '\ \\text{tal que}\ \\boldsymbol{x}= '\
          + latex(self.solP) + '+' \
          + latex(Matrix(self.sgen)) \
          + '\\boldsymbol{p}\\right. \\right\\}' \
       
    def __repr__(self):
        """Muestra el Espacio Nulo de una matriz en su representación python"""
        return repr(self.solP) + ' + Combinaciones lineales de (' + repr(self.sgen) + ')'

    def _repr_html_(self):
        """Construye la representación para el entorno jupyter notebook"""
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para la solución de un Sistema Homogéneo"""
        if self.determinado:
            return '\\text{Tiene solución única:  }\\boldsymbol{x}=' + latex(self.solP) 
        else:
            return '\\text{Conjunto de vectores: }' + self.EcParametricas()
              
class SELGJ:
    def __init__(self, A, b, rep=0):
        """Resuelve un Sistema de Ecuaciones Lineales
    
        mediante eliminación Gauss-Jordan por columas en la matriz
        ampliada y muestra los pasos dados"""
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        A  = Matrix(A)      
        MA = A if b.esNulo() else Matrix( BlockMatrix([ [A, Matrix([-b])] ]) )
        BM = Matrix( BlockMatrix([ [MA], [I(MA.n)] ]) )
        filasA = list(range(1,A.m+1))
        filasE = columnasE = list(range(1,A.n+1))
                
        LA = ElimGJ ( MA )
        
        if not b.esNulo() and not (filasA|LA|0).esNulo():
            self.tex = tex( {A.m, A.m+A.n} | BM | {A.n}, LA.pasos )
            raise ArithmeticError('No hay solución: Sistema incompatible')            
        EA        = I(MA.n) & T(LA.pasos[1])
        Normaliza = T([])    if b.esNulo() else T([(Fraction(1,0|EA|0),MA.n)])
        EA        = EA & Normaliza
        self.solP = V0(MA.n) if b.esNulo() else filasE|EA|0

        E                = filasE| EA |columnasE
        base             = [ (E|j) for j in columnasE if (filasA|LA|j).esNulo()]
        self.determinado = (len(base) == 0)
        self.sgen        = Sistema([V0(A.n)]) if self.determinado else Sistema(base)
        self.eafin       = EAfin(self.sgen,self.solP)

        self.pasos       = [ [], LA.pasos[1] + [Normaliza] ]
        self.tex         = tex( {A.m, A.m+A.n} | BM | {A.n}, self.pasos )
    def EcParametricas(self):
        """Representación paramétrica del SubEspacio"""
        return '\\left\\{ \\boldsymbol{x}\\in\\mathbb{R}^' \
          + latex(self.eafin.Rn) \
          + '\ \\left|\ \\exists\\boldsymbol{p}\\in\\mathbb{R}^' \
          + latex(len(self.sgen)) \
          + '\ \\text{tal que}\ \\boldsymbol{x}= '\
          + latex(self.solP) + '+' \
          + latex(Matrix(self.sgen)) \
          + '\\boldsymbol{p}\\right. \\right\\}' \
       
    def __repr__(self):
        """Muestra el Espacio Nulo de una matriz en su representación python"""
        return repr(self.solP) + ' + Combinaciones lineales de (' + repr(self.sgen) + ')'

    def _repr_html_(self):
        """Construye la representación para el entorno jupyter notebook"""
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para la solución de un Sistema Homogéneo"""
        if self.determinado:
            return '\\text{Tiene solución única:  }\\boldsymbol{x}=' + latex(self.solP) 
        else:
            return '\\text{Conjunto de vectores: }' + self.EcParametricas()
              
class Diagonaliza(Matrix):
    def __init__(self, A, espectro, Rep=0):
        """Diagonaliza por bloques triangulares la Matrix cuadrada A

        Encontrando una matriz semejante mediante trasformaciones de sus
        columnas y las transformaciones inversas espejo de las filas.

        espectro es el conjunto con los autovalores de la matriz.
        """
        def tex(data, pasos, TexPasosPrev=[]):
            def PasosYEscritura(data, pasos, TexPasosPrev=[]):
                """Escribe en LaTeX los pasos efectivos dados"""
                A   = Matrix(data);  p   = [[],[]]
                tex = latex(data) if len(TexPasosPrev)==0 else TexPasosPrev
                for l in range(0,2):
                    p[l] = [ T([j for j in pasos[l][i].t if (isinstance(j,set) and len(j)>1)   \
                                        or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)     \
                                        or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])  \
                                                                for i in range(0,len(pasos[l])) ]
                    p[l]   = [ t for t in p[l] if len(t.t)!=0]  # quitamos abreviaturas vacías     
                    if l==0:
                        for i in reversed(range(0,len(p[l]))):
                            tex += '\\xrightarrow[' + latex(p[l][i]) + ']{}'
                            if isinstance (data, Matrix):
                                         tex += latex( p[l][i] & A )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(p[l][i] & A)|key(data.ln) )
                    if l==1:
                        for i in range(0,len(p[l])):
                            tex += '\\xrightarrow{' + latex(p[l][i]) + '}'
                            if isinstance (data, Matrix):
                                         tex += latex( A & p[l][i] )
                            elif isinstance (data, BlockMatrix):
                                         tex += latex( key(data.lm)|(A & p[l][i])|key(data.ln) )
                return tex
            tex     = PasosYEscritura(data, pasos, TexPasosPrev)
            if rep:  
                from IPython.display import display, Math
                display(Math(tex))
            return tex
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]
            p = ppivote(self, r)
            while p in columnaOcupada:
                p = ppivote(self, p)
            return p
        D            = Matrix(A)
        S            = I(A.n)
        espectro     = list(espectro);         #espectro.sort()
        Tex          = latex( BlockMatrix( [[D], [S]] ) )
        pasosPrevios = [[],[]]
        selecc       = list(range(1,D.n+1))
        rep=0
        for l in espectro:
            m = selecc[-1]
            D = D-(l*I(D.n))
            Tex += '\\xrightarrow[' + latex(l) + '\\mathbf{I}]{(-)}' \
                                    + latex(BlockMatrix( [[D], [S]] ))
            TrCol = ElimG(selecc|D|selecc).pasos[1]
            pasos = [ [], TrCol ]
            pasosPrevios[1] = pasosPrevios[1] + pasos[1]

            Tex = tex( BlockMatrix( [[D], [S]] ), pasos, Tex)
            D = D & T(pasos[1])
            S = S & T(pasos[1])

            pasos = [ [T(pasos[1]).espejo()**-1] , []]
            pasosPrevios[0] = pasos[0] + pasosPrevios[0]

            Tex = tex( BlockMatrix( [[D], [S]] ), pasos, Tex)
            D = T(pasos[0]) & D

            if m < A.n:
                transf = []; columnaOcupada = set(selecc)
                for i in range(m,A.n+1):
                    p = BuscaNuevoPivote(i|D);
                    if p:
                        TrCol = [ T([(-Fraction(i|D|m, i|D|p), p, m)]) ]
                        pasos = [ [], TrCol ]
                        pasosPrevios[1] = pasosPrevios[1] + pasos[1]

                        Tex = tex( BlockMatrix( [[D], [S]] ), pasos, Tex)
                        D = D & T(pasos[1])
                        S = S & T(pasos[1])

                        pasos = [ [T(pasos[1]).espejo()**-1] , []]
                        pasosPrevios[0] = pasos[0] + pasosPrevios[0]

                        Tex = tex( BlockMatrix( [[D], [S]] ), pasos, Tex)
                        D = T(pasos[0]) & D

                        columnaOcupada.add(p)                        
            D = D+(l*I(D.n))
            Tex += '\\xrightarrow[' + latex(l) + '\\mathbf{I}]{(+)}' \
                                    + latex(BlockMatrix( [[D], [S]] ))
            
            selecc.pop()
            
        if Rep:
            from IPython.display import display, Math
            display(Math(Tex))
            
        espectro.sort(reverse=True)                
        self.espectro = espectro
        self.tex = Tex
        self.S   = S
        super(self.__class__ ,self).__init__(D.sis)
                   
class Normal(Matrix):
    def __init__(self, data):
        """Escalona por Gauss obteniendo una matriz cuyos pivotes son unos"""
        pivote=lambda v,k=0:([i for i,c in enumerate(v.sis,1)if(c!=0 and i>k)]+[0])[0]
        
        A = Matrix(data); r = 0
        self.rank = []
        for i in range(1,A.n+1):
            p = pivote((i|A),r)
            if p > 0:
                r += 1
                A & T( {p, r} )
                A & T( (1/Fraction(i|A|r), r) )
                A & T( [ (-(i|A|k), r, k) for k in range(r+1,A.n+1)] )

            self.rank+=[r]
              
        super(self.__class__ ,self).__init__(A.sis)        
def homogenea(A):
     """Devuelve una BlockMatriz con la solución del problema homogéneo"""
     stack=Matrix(BlockMatrix([[A],[I(A.n)]]))
     soluc=Normal(stack)
     col=soluc.rank[A.m-1]
     return {A.m} | soluc | {col}
