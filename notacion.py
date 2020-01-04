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
            L = ECL(self)

            if L.rank < L.n:
                raise ArithmeticError('Matrix singular')

            return Matrix( I(L.n) & T(ECUN(L).pasos[1]) )
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


def ppivote(v, k=0):
    """
    Devuelve el primer índice(i) mayor que k de un coeficiente(c) no 
    nulo del Vector v. En caso de no existir devuelve 0
    """            
    return ( [i for i,c in enumerate(v.sis, 1) if (c!=0 and i>k)] + [0] )[0]

class GaussCL(Matrix):
    def __init__(self, data):
        """Escalona una Matrix con eliminación por columnas (transf. Gauss)"""
        A = Matrix(data)
        r = 0
        for i in range(1, A.m+1):
           p = ppivote( i|A , r )
           if p > 0:
              r += 1
              A & T( {p, r} )
              A & T( [(Fraction(-(i|A|j),(i|A|r)), r, j) for j in range(r+1,A.n+1)] )
              
        super(self.__class__ ,self).__init__(A.sis)

class GaussCLsd(Matrix):
    def __init__(self, data):
        """Escalona una Matrix con eliminación por columnas (sin div.)"""
        A = Matrix(data)
        r = 0
        for i in range(1, A.m+1):
           p = ppivote( i|A , r )
           if p > 0:
              r += 1
              A & T( {p, r} )
              A & T([ T([ ( Fraction((i|A|j),(i|A|r)).denominator,  j),   \
                          (-Fraction((i|A|j),(i|A|r)).numerator, r, j) ]) \
                                                       for j in range(r+1,A.n+1) ])

        super(self.__class__ ,self).__init__(A.sis)        


class ECL(Matrix):
    def __init__(self, data, rep=0):
        """Escalona una Matrix con eliminación por columnas (transf. Gauss)"""
        A = Matrix(data); r = 0; pasos = [[], []]
        for i in range(1, A.m+1):
            p = ppivote( i|A , r )
            if p > 0:
                r += 1          
                Tr = T([ {p, r} ])
                pasos[1] += [Tr]  if Tr.t else []
                A & T( Tr )
                
                Tr = T([(Fraction(-(i|A|j), i|A|r), r, j) for j in range(r+1, A.n+1)])
                pasos[1] += [Tr]  if Tr.t else []
                A & T( Tr )
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)

class ECLsd(Matrix):
    def __init__(self, data, rep=0):
        """Escalona por eliminación por columnas (sin divisiones)"""
        A = Matrix(data); r = 0; pasos = [[], []]
        for i in range(1,A.m+1):
           p = ppivote((i|A),r)
           if p > 0:
               r += 1
               Tr = T( [ {p, r} ] )
               pasos[1] += [Tr]  if Tr.t else []
               A & T( Tr )
               
               Tr = T( [ T( [ ( Fraction((i|A|j),(i|A|r)).denominator,   j),    \
                              (-Fraction((i|A|j),(i|A|r)).numerator,  r, j) ] ) \
                                                    for j in range(r+1,A.n+1) ] )
               pasos[1] += [Tr]  if Tr.t else []
               A & T( Tr )
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
        
class ECLN(Matrix):
    def __init__(self, data, rep=0):
        """Escalona por eliminación por columnas haciendo pivotes unitarios"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A = Matrix(data); r = 0; pasos = [[], []]
        for i in range(1, A.m+1):
           p = ppivote( i|A , r )
           if p > 0:
              r += 1
              Tr = T( [ (Fraction(1, i|A|p), p) ] )
              pasos[1] += [Tr]  if Tr.t else []
              A & T( Tr )
              
              Tr = T( [ {p, r} ] )
              pasos[1] += [Tr]  if Tr.t else []
              A & T( Tr )
              
              Tr = T( [(-(i|A|j), r, j) for j in range(r+1,A.n+1)] )
              pasos[1] += [Tr]  if Tr.t else []
              A & T( Tr )
              
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
        
class ECUN(Matrix):
    def __init__(self, data, rep=0):
        """Escalona por eliminación por columnas haciendo pivotes unitarios"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A = Matrix(data); r = 0; pasos = [[], []]
        for i in reversed(range(1,A.m+1)):
           p = ppivote(reversed(i|A), r)
           if p > 0:
              r += 1
              Tr = T( [ (Fraction(1,(i|A|(A.n-p+1))), A.n-p+1 ) ] )
              pasos[1] += [Tr]  if Tr.t else []
              A & T( Tr )
              
              Tr = T( [ {A.n-p+1, A.n-r+1} ] )
              pasos[1] += [Tr]  if Tr.t else []
              A & T( Tr )
              
              Tr = T( [(-(i|A|j), A.n-r+1, j) for j in reversed(range(1,A.n-r+1)) ] )
              pasos[1] += [Tr]  if Tr.t else []
              A & T( Tr )
              
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
        
class ECU(Matrix):
    def __init__(self, data, rep=0):
        """Escalona una Matrix con eliminación por columnas (transf. Gauss)"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A = Matrix(data); r = 0; pasos = [[], []]
        for i in reversed(range(1,A.m+1)):
           p = ppivote(reversed(i|A), r)
           if p > 0:
              r += 1          
              Tr = T( [ {A.n-p+1, A.n-r+1} ] )
              pasos[1] += [Tr] 
              A & T( Tr );            
              Tr = T([ (Fraction(-(i|A|j), (i|A|(A.n-r+1))), A.n-r+1, j) \
                                     for j in reversed(range(1,A.n-r+1)) ] )
              pasos[1] += [Tr] if Tr.t else []
              A & T( Tr )
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
        
class ECUsd(Matrix):
    def __init__(self, data, rep=0):
        """Escalona una Matrix con eliminación por columnas (transf. Gauss)"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A = Matrix(data); r = 0; pasos = [[], []]
        for i in reversed(range(1,A.m+1)):
           p = ppivote(reversed(i|A), r)
           if p > 0:
              r += 1;  Tr = T( [ {A.n-p+1, A.n-r+1} ] );  pasos[1] += [Tr] 
              A & T( Tr )
              Tr = T( [ T( \
                   [ ( Fraction((i|A|j),(i|A|(A.n-r+1))).denominator, j),         \
                     (-Fraction((i|A|j),(i|A|(A.n-r+1))).numerator, A.n-r+1, j) ] \
                         ) for j in reversed(range(1,A.n-r+1)) ] )
              pasos[1] += [Tr] if Tr.t else []
              A & T( Tr )
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
        
class ECdcha(Matrix):
    def __init__(self, data, rep=0):
        """Reduce una Matrix con eliminación por columnas (transf. Gauss)"""
        A = Matrix(data); r = 0; pasos = [[], []];
        col = set()
        for i in range(1, A.m+1):
            p = ppivote( i|A )
            while p in col:
                p = ppivote( i|A, p)
            if p > 0:
                r += 1
                Tr = T([(Fraction(-(i|A|j),(i|A|p)), p, j) for j in \
                                             [x for x in range(p+1, A.n+1) if x!=p]])
                pasos[1] += [Tr]  if Tr.t else []
                A & T( Tr )
                col.add(p)
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)

class ECizdadcha(Matrix):
    def __init__(self, data, rep=0):
        """Reduce una Matrix con eliminación por columnas (transf. Gauss)"""
        A = Matrix(data); r = 0; pasos = [[], []];
        col = set()
        for i in range(1, A.m+1):
            p = ppivote( i|A )
            while p in col:
                p = ppivote( i|A, p)
            if p > 0:
                r += 1
                Tr = T([(Fraction(-(i|A|j),(i|A|p)), p, j) for j in \
                                              [x for x in range(1, A.n+1) if x!=p]])
                pasos[1] += [Tr]  if Tr.t else []
                A & T( Tr )
                col.add(p)
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)

class ECizdadchasd(Matrix):
    def __init__(self, data, rep=0):
        """Reduce una Matrix con eliminación por columnas (sin dividir)"""
        A = Matrix(data); r = 0; pasos = [[], []];
        col = set()
        for i in range(1, A.m+1):
            p = ppivote( i|A )
            while p in col:
                p = ppivote( i|A, p)
            if p > 0:
                r += 1
                Tr = T( [ T( [ ( Fraction((i|A|j),(i|A|p)).denominator,  j),    \
                               (-Fraction((i|A|j),(i|A|p)).numerator, p, j) ] ) \
                                       for j in [x for x in range(1,A.n+1) if x!=p] ] )
                pasos[1] += [Tr]  if Tr.t else []
                A & T( Tr )
                col.add(p)
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)

class ECdchasd(Matrix):
    def __init__(self, data, rep=0):
        """Reduce una Matrix con eliminación por columnas (sin dividir)"""
        A = Matrix(data); r = 0; pasos = [[], []];
        col = set()
        for i in range(1, A.m+1):
            p = ppivote( i|A )
            while p in col:
                p = ppivote( i|A, p)
            if p > 0:
                r += 1
                Tr = T( [ T( [ ( Fraction((i|A|j),(i|A|p)).denominator,  j),    \
                               (-Fraction((i|A|j),(i|A|p)).numerator, p, j) ] ) \
                                   for j in [x for x in range(p,A.n+1) if x!=p] ] )
                pasos[1] += [Tr]  if Tr.t else []
                A & T( Tr )
                col.add(p)
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)

class NormDiag(Matrix):
    def __init__(self, data, rep=0):
        """Normaliza a uno los componentes no nulos de la diagonal principal"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A     = Matrix(data);   pasos = [[],[]];
        Tr = T([ (Fraction(1,(j|A|j)), j) for j in range(1,A.n+1) if (j|A|j)!=0 ] )
        pasos[1] = [Tr] if Tr.t else []
        A & T( [Tr] )
        PPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = PasosYEscritura(data, pasos, TexPPrev)
        if rep:  
                from IPython.display import display, Math
                display(Math(self.tex))
        self.rank  = data.rank if hasattr(data, 'rank') and data.rank else []
        pasos[1] = PPrevios[1] + pasos[1]
        self.pasos = pasos 
        super(self.__class__ ,self).__init__(A.sis)

class EFU(Matrix):
    def __init__(self, data, rep=0):
        """Escalona una Matrix con eliminación por filas (transf. Gauss)"""
        A = Matrix(data); r = 0; pasos = [[], []]
        for j in range(1,A.n+1):
           p = ppivote((A|j),r)
           if p > 0:
              r += 1          
              Tr = T( [ {p, r} ] );  pasos[0] += [Tr] 
              T( Tr ) & A             
              Tr = T( [(Fraction(-(i|A|j),(r|A|j)), r, i) for i in range(r+1,A.m+1)] ) 
              pasos[0] += list(reversed([Tr])) 
              T( Tr ) & A
              pasos[0]=list(reversed(pasos[0]))
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
        
class EFUN(Matrix):
    def __init__(self, data, rep=0):
        """Escalona con eliminación por filas haciendo pivotes unitarios"""
        A = Matrix(data); r = 0; pasos = [[], []]
        for j in range(1,A.n+1):
           p = ppivote((A|j),r)
           if p > 0:
              r += 1          
              Tr = T( [ {p, r} ] );                    pasos[0] += [Tr] 
              T( Tr ) & A             
              Tr = T( [ (Fraction(1,(j|A|r)), r) ] );  pasos[0] += [Tr] 
              T( Tr ) & A
              Tr = T( [(-(i|A|j), r, i) for i in range(r+1,A.m+1)] )
              pasos[0] += list(reversed([Tr]))  if Tr.t else []
              T( Tr ) & A
        pasos[0]=list(reversed(pasos[0]))
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
        
class EFL(Matrix):
    def __init__(self, data, rep=0):
        """Escalona una Matrix con eliminación por filas (L)"""
        A = Matrix(data); r = 0; pasos = [[], []]
        for j in reversed(range(1,A.n+1)):
           p = ppivote(reversed(A|j),r)
           if p > 0:
              r += 1          
              Tr = T( [ {A.m-p+1, A.m-r+1} ] );       pasos[0] += [Tr] 
              T( Tr ) & A
              Tr = T([ ( Fraction(-(i|A|j), (A.m-r+1)|A|j), A.m-r+1, i) \
                                                for i in range(1, A.m-r+1) ] )
              pasos[0] += [Tr] if Tr.t else []
              T( Tr ) & A 
        pasos[0]=list(reversed(pasos[0]))
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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

        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex     = PasosYEscritura(data, pasos, TexPasosPrev)
        if rep:  
            from IPython.display import display, Math
            display(Math(self.tex))
        self.rank  = r
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 

        super(self.__class__ ,self).__init__(A.sis)
class InvMat(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las columnas"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A     = Matrix(data)

        if A.m != A.n:
            raise ValueError('Matrix no cuadrada')

        L = ECL(A)
        if L.rank < A.n:
            raise ArithmeticError('Matrix singular')

        M  = ECUN(L)
        stack = BlockMatrix([[A],[I(A.n)]])
        self.tex   = PasosYEscritura(stack, M.pasos)
        if rep:
           from IPython.display import display, Math
           display(Math(self.tex))

        Inv        = I(A.n) & T(M.pasos[1])  
        self.pasos = M.pasos 
        super(self.__class__ ,self).__init__(Inv.sis)

class InvMatF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A     = Matrix(data)
        Id    = EFUN(EFL(A))
        stack = BlockMatrix([[A,I(A.m)]])
        self.tex   = PasosYEscritura(stack, Id.pasos)
        if rep:
           from IPython.display import display, Math
           display(Math(self.tex))

        Inv        = T(Id.pasos[0]) & I(A.n)   
        self.pasos = Id.pasos 
        super(self.__class__ ,self).__init__(Inv.sis)

class InvMatFC(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas y columnas"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A     = Matrix(data)
        Id    = ECLN(EFU(A))
        stack = BlockMatrix([[A,I(A.m)],[I(A.n),M0(A.m,A.n)]])
        self.tex   = PasosYEscritura(stack, Id.pasos)
        if rep:
           from IPython.display import display, Math
           display(Math(self.tex))

        Inv        = ( I(A.n) & T(Id.pasos[1]) ) * ( T(Id.pasos[0]) & I(A.n) )
        self.pasos = Id.pasos 
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
        if not isinstance(data, (Sistema, Matrix)):
             raise ValueError(' Argumento debe ser un Sistema o Matrix ')
        if isinstance(data, Sistema):
             A          = Matrix(data)
             L          = ECLsd(A)
             self.dim   = L.rank
             self.base  = Sistema([L|j for j in range(1,L.rank+1)])
             self.sgen  = self.base if L.rank else Sistema([V0(A.m)])
             self.cart  = ~Matrix(Homogenea(~A).sgen)
             self.Rn    = A.m
        if isinstance(data, Matrix):
             A          = data
             H          = Homogenea(A)
             self.sgen  = H.sgen  
             self.dim   = 0 if H.determinado else len(self.sgen)
             self.base  = self.sgen if self.dim else Sistema([])
             self.cart  = ~Matrix(Homogenea(~Matrix(self.sgen)).sgen)
             self.Rn    = A.n
    def contenidoEn(self, other):
        """Indica si este SubEspacio está contenido en other"""
        self.verificacion(other)
        if isinstance(other, SubEspacio):
            #return all ([(other.cart*w).esNulo() for w in self.sgen])
            M = Matrix(BlockMatrix([[Matrix(other.sgen), Matrix( self.sgen)]]))
            return True if (ECLsd(M).rank == other.dim) else False
        else:
            return other.v.esNulo() and self.contenidoEn(other.S)
        
    def __eq__(self, other):
        """Indica si un subespacio de Rn es igual a otro"""
        self.verificacion(other)
        return self.contenidoEn(other) and other.contenidoEn(self)

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
        self.S = data if isinstance(data, SubEspacio) else SubEspacio(data)
        if not isinstance(v, Vector) or v.n != self.S.Rn:
             raise ValueError('v y SubEspacio deben estar en el mismo espacio vectorial')
        MA     = Matrix( BlockMatrix([ [ Matrix(self.S.sgen), Matrix([v]) ] ]) )
        self.v = ECdcha( MA )|0
        self.Rn=self.S.Rn
        
    def __contains__(self, other):
        """Indica si un Vector pertenece a un EAfin"""
        if not isinstance(other, Vector) or other.n != self.S.cart.n:
            raise ValueError('Es necesario un Vector con el número adecuado de componentes')
        return (self.S.cart)*other == (self.S.cart)*self.v

    def contenidoEn(self, other):
        """Indica si este EAfin está contenido en other"""
        self.verificacion(other)
        if isinstance(other, SubEspacio):
             return self.v in other and self.S.contenidoEn(other)
        else:
             return self.v in other and self.S.contenidoEn(other.S)

    def __eq__(self, other):
        """Indica si un EAfin de Rn es igual a other"""
        self.verificacion(other)
        return self.contenidoEn(other) and other.contenidoEn(self)

    def __ne__(self, other):
        """Indica si un subespacio de Rn es distinto de other"""
        self.verificacion(other)
        return not (self == other)

    def verificacion(self,other):
        if not isinstance(other, (SubEspacio, EAfin)) or  not self.Rn == other.Rn: 
            raise \
             ValueError('Ambos argumentos deben ser subconjuntos de en un mismo espacio')

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
        """Describe el espacio nulo de una matriz y los pasos para encontrarlo"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        A                = Matrix(data)
        L                = ECLsd(A)
        E                = I(A.n) & T(L.pasos[1])

        base             = [Vector(E|j) for j in range(L.rank+1, L.n+1)]
        dim              = len(base)
        self.determinado = (dim == 0)
        self.sgen        = Sistema(base) if dim else Sistema([V0(A.n)])

        self.tex         = PasosYEscritura(BlockMatrix([[A],[I(A.n)]]), L.pasos)
        if rep:
           from IPython.display import display, Math
           display(Math(self.tex))
       
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
    
        y muestra los pasos para encontrarlo"""
        def PasosYEscritura(data,pasos,TexPasosPrev=[]):
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
        MA               = Matrix(BlockMatrix([ [A, -Matrix([b]) ] ] ))

        R                = ECLsd( A )
        E                = I(A.n) & T(R.pasos[1])
        base             = [Vector(E|j) for j in range(R.rank+1, R.n+1)]
        dim              = len(base)
        self.determinado = (dim == 0)
        self.enulo       = Sistema(base) if dim else Sistema([V0(A.n)])

        RA               = ECdchasd( Matrix(MA) & T(R.pasos[1]) )
        if not (RA|0).esNulo():
            if rep:
                pasos = [ [], R.pasos[1] + RA.pasos[1] ]
                BM       = Matrix(BlockMatrix([[MA],[I(MA.n)]]))
                self.tex = PasosYEscritura( {A.m,A.m+A.n}|BM|{A.n}, pasos)
                from IPython.display import display, Math
                display(Math(self.tex))
            raise ArithmeticError('No hay solución: Sistema incompatible')
        
        EA               = I(MA.n) & T(R.pasos[1]) & T(RA.pasos[1])
        Normalizacion    = T([(Fraction(1, 0|EA|0 ), MA.n)])
        EA               = EA & Normalizacion
        self.solP        = list(range(1,A.n+1))|EA|0 

        pasos = [ [], R.pasos[1] + RA.pasos[1] + [Normalizacion] ]
        BM    = Matrix(BlockMatrix([[MA],[I(MA.n)]]))
        self.tex = PasosYEscritura( {A.m,A.m+A.n}|BM|{A.n}, pasos)
        if rep:
           from IPython.display import display, Math
           display(Math(self.tex))

        self.eafin=EAfin(self.enulo,self.solP)
   
    def EcParametricas(self):
        """Representación paramétrica del SubEspacio"""
        return '\\left\\{ \\boldsymbol{x}\\in\\mathbb{R}^' \
          + latex(self.eafin.Rn) \
          + '\ \\left|\ \\exists\\boldsymbol{a}\\in\\mathbb{R}^' \
          + latex(len(self.enulo)) \
          + '\ \\text{tal que}\ \\boldsymbol{x}= '\
          + latex(self.solP) + '+' \
          + latex(Matrix(self.enulo)) \
          + '\\boldsymbol{a}\\right. \\right\\}' \
       
    #def __repr__(self):
    #    """Muestra el Espacio Nulo de una matriz en su representación python"""
    #    return 'Combinaciones lineales de (' + repr(self.sol) + ')'

    def _repr_html_(self):
        """Construye la representación para el entorno jupyter notebook"""
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para la solución de un Sistema Homogéneo"""
        if self.determinado:
            return '\\text{Tiene solución única:  }\\boldsymbol{x}=' + latex(self.solP) 
        else:
            return '\\text{Conjunto de vectores: }' + self.EcParametricas()
           
class Normal(Matrix):
    def __init__(self, data):
        """Escalona por Gauss obteniendo una matriz cuyos pivotes son unos"""
        A = Matrix(data); r = 0
        self.rank = []
        for i in range(1,A.n+1):
           p = ppivote((i|A),r)
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
