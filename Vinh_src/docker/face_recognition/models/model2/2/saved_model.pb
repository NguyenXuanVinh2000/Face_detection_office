??
?!?!
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( ?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?
&
	ZerosLike
x"T
y"T"	
Ttype"train*1.15.02v1.15.0-rc3-22-g590d6ee8Ԗ
r
dense_1_inputPlaceholder*
dtype0*
shape:??????????*(
_output_shapes
:??????????
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"?      *
_output_shapes
:*!
_class
loc:@dense_1/kernel*
dtype0
?
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓?*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0
?
-dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *:͓=*!
_class
loc:@dense_1/kernel*
dtype0
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
??
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
??*!
_class
loc:@dense_1/kernel
?
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
??*
T0*!
_class
loc:@dense_1/kernel
?
dense_1/kernelVarHandleOp*
shape:
??*
shared_namedense_1/kernel*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
?
.dense_1/bias/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense_1/bias*
_output_shapes
:*
dtype0*
valueB:?
?
$dense_1/bias/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
?
dense_1/bias/Initializer/zerosFill.dense_1/bias/Initializer/zeros/shape_as_tensor$dense_1/bias/Initializer/zeros/Const*
_output_shapes	
:?*
_class
loc:@dense_1/bias*
T0
?
dense_1/biasVarHandleOp*
dtype0*
_output_shapes
: *
_class
loc:@dense_1/bias*
shape:?*
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
n
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
y
dense_1/MatMulMatMuldense_1_inputdense_1/MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0
h
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
}
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*(
_output_shapes
:??????????*
T0
X
dense_1/ReluReludense_1/BiasAdd*(
_output_shapes
:??????????*
T0
[
dropout_1/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
S
dropout_1/dropout/ShapeShapedense_1/Relu*
_output_shapes
:*
T0
i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$dropout_1/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniformdropout_1/dropout/Shape*
dtype0*(
_output_shapes
:??????????*
T0
?
$dropout_1/dropout/random_uniform/subSub$dropout_1/dropout/random_uniform/max$dropout_1/dropout/random_uniform/min*
_output_shapes
: *
T0
?
$dropout_1/dropout/random_uniform/mulMul.dropout_1/dropout/random_uniform/RandomUniform$dropout_1/dropout/random_uniform/sub*(
_output_shapes
:??????????*
T0
?
 dropout_1/dropout/random_uniformAdd$dropout_1/dropout/random_uniform/mul$dropout_1/dropout/random_uniform/min*
T0*(
_output_shapes
:??????????
\
dropout_1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
n
dropout_1/dropout/subSubdropout_1/dropout/sub/xdropout_1/dropout/rate*
T0*
_output_shapes
: 
`
dropout_1/dropout/truediv/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
y
dropout_1/dropout/truedivRealDivdropout_1/dropout/truediv/xdropout_1/dropout/sub*
_output_shapes
: *
T0
?
dropout_1/dropout/GreaterEqualGreaterEqual dropout_1/dropout/random_uniformdropout_1/dropout/rate*
T0*(
_output_shapes
:??????????
x
dropout_1/dropout/mulMuldense_1/Reludropout_1/dropout/truediv*
T0*(
_output_shapes
:??????????
?
dropout_1/dropout/CastCastdropout_1/dropout/GreaterEqual*(
_output_shapes
:??????????*

DstT0*

SrcT0

?
dropout_1/dropout/mul_1Muldropout_1/dropout/muldropout_1/dropout/Cast*(
_output_shapes
:??????????*
T0
?
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
dtype0*
valueB"      *
_output_shapes
:
?
-dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *׳]?*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
?
-dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *׳]=*
dtype0*!
_class
loc:@dense_2/kernel
?
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_2/kernel*
T0*
dtype0* 
_output_shapes
:
??
?
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_2/kernel
?
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
??*
T0
?
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
??*
T0
?
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shared_namedense_2/kernel*!
_class
loc:@dense_2/kernel*
shape:
??
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
q
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
?
.dense_2/bias/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense_2/bias*
valueB:?*
_output_shapes
:*
dtype0
?
$dense_2/bias/Initializer/zeros/ConstConst*
_class
loc:@dense_2/bias*
_output_shapes
: *
valueB
 *    *
dtype0
?
dense_2/bias/Initializer/zerosFill.dense_2/bias/Initializer/zeros/shape_as_tensor$dense_2/bias/Initializer/zeros/Const*
_output_shapes	
:?*
_class
loc:@dense_2/bias*
T0
?
dense_2/biasVarHandleOp*
shape:?*
dtype0*
_class
loc:@dense_2/bias*
shared_namedense_2/bias*
_output_shapes
: 
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
b
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
dtype0
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:?
n
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
??
?
dense_2/MatMulMatMuldropout_1/dropout/mul_1dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????
h
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:?
}
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*(
_output_shapes
:??????????*
T0
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:??????????*
T0
[
dropout_2/dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
S
dropout_2/dropout/ShapeShapedense_2/Relu*
_output_shapes
:*
T0
i
$dropout_2/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
i
$dropout_2/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniformdropout_2/dropout/Shape*(
_output_shapes
:??????????*
T0*
dtype0
?
$dropout_2/dropout/random_uniform/subSub$dropout_2/dropout/random_uniform/max$dropout_2/dropout/random_uniform/min*
T0*
_output_shapes
: 
?
$dropout_2/dropout/random_uniform/mulMul.dropout_2/dropout/random_uniform/RandomUniform$dropout_2/dropout/random_uniform/sub*
T0*(
_output_shapes
:??????????
?
 dropout_2/dropout/random_uniformAdd$dropout_2/dropout/random_uniform/mul$dropout_2/dropout/random_uniform/min*(
_output_shapes
:??????????*
T0
\
dropout_2/dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
n
dropout_2/dropout/subSubdropout_2/dropout/sub/xdropout_2/dropout/rate*
T0*
_output_shapes
: 
`
dropout_2/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
y
dropout_2/dropout/truedivRealDivdropout_2/dropout/truediv/xdropout_2/dropout/sub*
T0*
_output_shapes
: 
?
dropout_2/dropout/GreaterEqualGreaterEqual dropout_2/dropout/random_uniformdropout_2/dropout/rate*
T0*(
_output_shapes
:??????????
x
dropout_2/dropout/mulMuldense_2/Reludropout_2/dropout/truediv*(
_output_shapes
:??????????*
T0
?
dropout_2/dropout/CastCastdropout_2/dropout/GreaterEqual*

DstT0*

SrcT0
*(
_output_shapes
:??????????
?
dropout_2/dropout/mul_1Muldropout_2/dropout/muldropout_2/dropout/Cast*(
_output_shapes
:??????????*
T0
?
/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"   (   *!
_class
loc:@dense_3/kernel*
dtype0
?
-dense_3/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *?ʙ?
?
-dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
valueB
 *?ʙ=
?
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	?(*!
_class
loc:@dense_3/kernel*
T0
?
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
T0
?
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?(*
T0*!
_class
loc:@dense_3/kernel
?
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes
:	?(
?
dense_3/kernelVarHandleOp*
dtype0*!
_class
loc:@dense_3/kernel*
shared_namedense_3/kernel*
_output_shapes
: *
shape:	?(
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
q
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*
dtype0
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?(*
dtype0
?
dense_3/bias/Initializer/zerosConst*
_output_shapes
:(*
_class
loc:@dense_3/bias*
dtype0*
valueB(*    
?
dense_3/biasVarHandleOp*
shape:(*
_output_shapes
: *
dtype0*
shared_namedense_3/bias*
_class
loc:@dense_3/bias
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
dtype0
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:(*
dtype0
m
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?(*
dtype0
?
dense_3/MatMulMatMuldropout_2/dropout/mul_1dense_3/MatMul/ReadVariableOp*'
_output_shapes
:?????????(*
T0
g
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:(
|
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????(
]
dense_3/SoftmaxSoftmaxdense_3/BiasAdd*'
_output_shapes
:?????????(*
T0
?
dense_3_targetPlaceholder*%
shape:??????????????????*
dtype0*0
_output_shapes
:??????????????????
v
total/Initializer/zerosConst*
_output_shapes
: *
_class

loc:@total*
dtype0*
valueB
 *    
x
totalVarHandleOp*
shape: *
_output_shapes
: *
shared_nametotal*
dtype0*
_class

loc:@total
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
v
count/Initializer/zerosConst*
_output_shapes
: *
_class

loc:@count*
dtype0*
valueB
 *    
x
countVarHandleOp*
_output_shapes
: *
shape: *
dtype0*
shared_namecount*
_class

loc:@count
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
g
metrics/acc/ArgMax/dimensionConst*
valueB :
?????????*
_output_shapes
: *
dtype0
x
metrics/acc/ArgMaxArgMaxdense_3_targetmetrics/acc/ArgMax/dimension*#
_output_shapes
:?????????*
T0
i
metrics/acc/ArgMax_1/dimensionConst*
_output_shapes
: *
valueB :
?????????*
dtype0
}
metrics/acc/ArgMax_1ArgMaxdense_3/Softmaxmetrics/acc/ArgMax_1/dimension*
T0*#
_output_shapes
:?????????
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*#
_output_shapes
:?????????*
T0	
h
metrics/acc/CastCastmetrics/acc/Equal*

DstT0*#
_output_shapes
:?????????*

SrcT0

[
metrics/acc/ConstConst*
dtype0*
valueB: *
_output_shapes
:
\
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*
_output_shapes
: *
T0
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
?
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
K
metrics/acc/SizeSizemetrics/acc/Cast*
_output_shapes
: *
T0
\
metrics/acc/Cast_1Castmetrics/acc/Size*

SrcT0*

DstT0*
_output_shapes
: 
?
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_1 ^metrics/acc/AssignAddVariableOp*
dtype0
?
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
?
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
?
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
?
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
_output_shapes
: *
T0
\
loss/dense_3_loss/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
z
8loss/dense_3_loss/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
x
9loss/dense_3_loss/softmax_cross_entropy_with_logits/ShapeShapedense_3/BiasAdd*
T0*
_output_shapes
:
|
:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :
z
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_1Shapedense_3/BiasAdd*
_output_shapes
:*
T0
{
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
?
7loss/dense_3_loss/softmax_cross_entropy_with_logits/SubSub:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_19loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub/y*
_output_shapes
: *
T0
?
?loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/beginPack7loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub*
T0*
N*
_output_shapes
:
?
>loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
?
9loss/dense_3_loss/softmax_cross_entropy_with_logits/SliceSlice;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_1?loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/begin>loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
?
Closs/dense_3_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
?????????*
_output_shapes
:*
dtype0
?
?loss/dense_3_loss/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
:loss/dense_3_loss/softmax_cross_entropy_with_logits/concatConcatV2Closs/dense_3_loss/softmax_cross_entropy_with_logits/concat/values_09loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice?loss/dense_3_loss/softmax_cross_entropy_with_logits/concat/axis*
_output_shapes
:*
T0*
N
?
;loss/dense_3_loss/softmax_cross_entropy_with_logits/ReshapeReshapedense_3/BiasAdd:loss/dense_3_loss/softmax_cross_entropy_with_logits/concat*0
_output_shapes
:??????????????????*
T0
|
:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
y
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_2Shapedense_3_target*
T0*
_output_shapes
:
}
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
?
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1Sub:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_2;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
?
Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1*
T0*
N*
_output_shapes
:
?
@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1Slice;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_2Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/begin@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/size*
T0*
Index0*
_output_shapes
:
?
Eloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Aloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
<loss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Eloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/values_0;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1Aloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/axis*
N*
T0*
_output_shapes
:
?
=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_3_target<loss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1*
T0*0
_output_shapes
:??????????????????
?
3loss/dense_3_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits;loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_1*?
_output_shapes-
+:?????????:??????????????????*
T0
}
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
?
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2Sub8loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
?
Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
?
@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2*
_output_shapes
:*
T0*
N
?
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2Slice9loss/dense_3_loss/softmax_cross_entropy_with_logits/ShapeAloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/begin@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/size*
T0*
Index0*
_output_shapes
:
?
=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape3loss/dense_3_loss/softmax_cross_entropy_with_logits;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2*
T0*#
_output_shapes
:?????????
k
&loss/dense_3_loss/weighted_loss/Cast/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Tloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
dtype0*
_output_shapes
: *
valueB 
?
Sloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
_output_shapes
: *
dtype0
?
Sloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:
?
Rloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
j
bloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
?
Aloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2c^loss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:
?
Aloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
;loss/dense_3_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:?????????*
T0
?
1loss/dense_3_loss/weighted_loss/broadcast_weightsMul&loss/dense_3_loss/weighted_loss/Cast/x;loss/dense_3_loss/weighted_loss/broadcast_weights/ones_like*#
_output_shapes
:?????????*
T0
?
#loss/dense_3_loss/weighted_loss/MulMul=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_21loss/dense_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:?????????
c
loss/dense_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
}
loss/dense_3_loss/SumSum#loss/dense_3_loss/weighted_loss/Mulloss/dense_3_loss/Const_1*
_output_shapes
: *
T0
l
loss/dense_3_loss/num_elementsSize#loss/dense_3_loss/weighted_loss/Mul*
T0*
_output_shapes
: 
{
#loss/dense_3_loss/num_elements/CastCastloss/dense_3_loss/num_elements*

DstT0*

SrcT0*
_output_shapes
: 
\
loss/dense_3_loss/Const_2Const*
valueB *
_output_shapes
: *
dtype0
q
loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/Sumloss/dense_3_loss/Const_2*
T0*
_output_shapes
: 
?
loss/dense_3_loss/valueDivNoNanloss/dense_3_loss/Sum_1#loss/dense_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/dense_3_loss/value*
_output_shapes
: *
T0
q
iter/Initializer/zerosConst*
value	B	 R *
_output_shapes
: *
_class
	loc:@iter*
dtype0	
u
iterVarHandleOp*
shared_nameiter*
_output_shapes
: *
_class
	loc:@iter*
dtype0	*
shape: 
Y
%iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpiter*
_output_shapes
: 
J
iter/AssignAssignVariableOpiteriter/Initializer/zeros*
dtype0	
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
j
'training/Adam/gradients/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
+training/Adam/gradients/gradients/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*
_output_shapes
: *
T0
?
3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/dense_3_loss/value*
_output_shapes
: *
T0
?
5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
_output_shapes
: *
T0
?
Dtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
?
Ftraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Ttraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ShapeFtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
Itraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1#loss/dense_3_loss/num_elements/Cast*
_output_shapes
: *
T0
?
Btraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/SumSumItraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nanTtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: 
?
Ftraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ReshapeReshapeBtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/SumDtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape*
T0*
_output_shapes
: 
?
Btraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/NegNegloss/dense_3_loss/Sum_1*
T0*
_output_shapes
: 
?
Ktraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_1DivNoNanBtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Neg#loss/dense_3_loss/num_elements/Cast*
_output_shapes
: *
T0
?
Ktraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_2DivNoNanKtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_1#loss/dense_3_loss/num_elements/Cast*
_output_shapes
: *
T0
?
Btraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ktraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
?
Dtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Sum_1SumBtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/mulVtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: 
?
Htraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Reshape_1ReshapeDtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Sum_1Ftraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/Shape_1*
T0*
_output_shapes
: 
?
Ltraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
Ftraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeReshapeFtraining/Adam/gradients/gradients/loss/dense_3_loss/value_grad/ReshapeLtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0
?
Dtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
?
Ctraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/TileTileFtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/ReshapeDtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/Const*
T0*
_output_shapes
: 
?
Jtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
?
Dtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/ReshapeReshapeCtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_1_grad/TileJtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Reshape/shape*
_output_shapes
:*
T0
?
Btraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/ShapeShape#loss/dense_3_loss/weighted_loss/Mul*
T0*
_output_shapes
:
?
Atraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/TileTileDtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/ReshapeBtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Shape*#
_output_shapes
:?????????*
T0
?
Ptraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/ShapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:
?
Rtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape_1Shape1loss/dense_3_loss/weighted_loss/broadcast_weights*
T0*
_output_shapes
:
?
`training/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/ShapeRtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
Ntraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/MulMulAtraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Tile1loss/dense_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:?????????
?
Ntraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/SumSumNtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Mul`training/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0
?
Rtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/ReshapeReshapeNtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/SumPtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape*
T0*#
_output_shapes
:?????????
?
Ptraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Mul_1Mul=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2Atraining/Adam/gradients/gradients/loss/dense_3_loss/Sum_grad/Tile*#
_output_shapes
:?????????*
T0
?
Ptraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Sum_1SumPtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Mul_1btraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0
?
Ttraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Reshape_1ReshapePtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Sum_1Rtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Shape_1*
T0*#
_output_shapes
:?????????
?
jtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape3loss/dense_3_loss/softmax_cross_entropy_with_logits*
T0*
_output_shapes
:
?
ltraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapeRtraining/Adam/gradients/gradients/loss/dense_3_loss/weighted_loss/Mul_grad/Reshapejtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*#
_output_shapes
:?????????
?
,training/Adam/gradients/gradients/zeros_like	ZerosLike5loss/dense_3_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:??????????????????
?
itraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
dtype0*
valueB :
?????????*
_output_shapes
: 
?
etraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeitraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/mulMuletraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims5loss/dense_3_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:??????????????????
?
etraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax;loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:??????????????????
?
^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/NegNegetraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax*0
_output_shapes
:??????????????????*
T0
?
ktraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
?????????*
_output_shapes
: *
dtype0
?
gtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapektraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:?????????
?
`training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/mul_1Mulgtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:??????????????????
?
htraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_3/BiasAdd*
T0*
_output_shapes
:
?
jtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshape^training/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits_grad/mulhtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*'
_output_shapes
:?????????(
?
Btraining/Adam/gradients/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGradjtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
_output_shapes
:(
?
<training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMulMatMuljtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_3/MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0*
transpose_b(
?
>training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul_1MatMuldropout_2/dropout/mul_1jtraining/Adam/gradients/gradients/loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
transpose_a(*
_output_shapes
:	?(*
T0
?
Dtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/ShapeShapedropout_2/dropout/mul*
T0*
_output_shapes
:
?
Ftraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Shape_1Shapedropout_2/dropout/Cast*
_output_shapes
:*
T0
?
Ttraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/ShapeFtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
Btraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/MulMul<training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMuldropout_2/dropout/Cast*
T0*(
_output_shapes
:??????????
?
Btraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/SumSumBtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/MulTtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
?
Ftraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/ReshapeReshapeBtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/SumDtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Shape*
T0*(
_output_shapes
:??????????
?
Dtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Mul_1Muldropout_2/dropout/mul<training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
Dtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Sum_1SumDtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Mul_1Vtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
Htraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Reshape_1ReshapeDtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Sum_1Ftraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Shape_1*
T0*(
_output_shapes
:??????????
~
Btraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/ShapeShapedense_2/Relu*
_output_shapes
:*
T0
?
Dtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Shape_1Shapedropout_2/dropout/truediv*
_output_shapes
: *
T0
?
Rtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/ShapeDtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
@training/Adam/gradients/gradients/dropout_2/dropout/mul_grad/MulMulFtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Reshapedropout_2/dropout/truediv*
T0*(
_output_shapes
:??????????
?
@training/Adam/gradients/gradients/dropout_2/dropout/mul_grad/SumSum@training/Adam/gradients/gradients/dropout_2/dropout/mul_grad/MulRtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
?
Dtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/ReshapeReshape@training/Adam/gradients/gradients/dropout_2/dropout/mul_grad/SumBtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Shape*(
_output_shapes
:??????????*
T0
?
Btraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Mul_1Muldense_2/ReluFtraining/Adam/gradients/gradients/dropout_2/dropout/mul_1_grad/Reshape*(
_output_shapes
:??????????*
T0
?
Btraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Sum_1SumBtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Mul_1Ttraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
Ftraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Reshape_1ReshapeBtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Sum_1Dtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Shape_1*
_output_shapes
: *
T0
?
<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGradReluGradDtraining/Adam/gradients/gradients/dropout_2/dropout/mul_grad/Reshapedense_2/Relu*
T0*(
_output_shapes
:??????????
?
Btraining/Adam/gradients/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGrad*
T0*
_output_shapes	
:?
?
<training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGraddense_2/MatMul/ReadVariableOp*(
_output_shapes
:??????????*
transpose_b(*
T0
?
>training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul_1MatMuldropout_1/dropout/mul_1<training/Adam/gradients/gradients/dense_2/Relu_grad/ReluGrad*
transpose_a(*
T0* 
_output_shapes
:
??
?
Dtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/ShapeShapedropout_1/dropout/mul*
T0*
_output_shapes
:
?
Ftraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Shape_1Shapedropout_1/dropout/Cast*
_output_shapes
:*
T0
?
Ttraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/ShapeFtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
Btraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/MulMul<training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMuldropout_1/dropout/Cast*
T0*(
_output_shapes
:??????????
?
Btraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/SumSumBtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/MulTtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0
?
Ftraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/ReshapeReshapeBtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/SumDtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Shape*
T0*(
_output_shapes
:??????????
?
Dtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Mul_1Muldropout_1/dropout/mul<training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
Dtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Sum_1SumDtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Mul_1Vtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
Htraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Reshape_1ReshapeDtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Sum_1Ftraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Shape_1*(
_output_shapes
:??????????*
T0
~
Btraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/ShapeShapedense_1/Relu*
_output_shapes
:*
T0
?
Dtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Shape_1Shapedropout_1/dropout/truediv*
_output_shapes
: *
T0
?
Rtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/ShapeDtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????
?
@training/Adam/gradients/gradients/dropout_1/dropout/mul_grad/MulMulFtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Reshapedropout_1/dropout/truediv*(
_output_shapes
:??????????*
T0
?
@training/Adam/gradients/gradients/dropout_1/dropout/mul_grad/SumSum@training/Adam/gradients/gradients/dropout_1/dropout/mul_grad/MulRtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0
?
Dtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/ReshapeReshape@training/Adam/gradients/gradients/dropout_1/dropout/mul_grad/SumBtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Shape*
T0*(
_output_shapes
:??????????
?
Btraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Mul_1Muldense_1/ReluFtraining/Adam/gradients/gradients/dropout_1/dropout/mul_1_grad/Reshape*(
_output_shapes
:??????????*
T0
?
Btraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Sum_1SumBtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Mul_1Ttraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
?
Ftraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Reshape_1ReshapeBtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Sum_1Dtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Shape_1*
T0*
_output_shapes
: 
?
<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGradReluGradDtraining/Adam/gradients/gradients/dropout_1/dropout/mul_grad/Reshapedense_1/Relu*(
_output_shapes
:??????????*
T0
?
Btraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGrad*
T0*
_output_shapes	
:?
?
<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMulMatMul<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGraddense_1/MatMul/ReadVariableOp*
T0*
transpose_b(*(
_output_shapes
:??????????
?
>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input<training/Adam/gradients/gradients/dense_1/Relu_grad/ReluGrad*
transpose_a(* 
_output_shapes
:
??*
T0
?
.training/Adam/beta_1/Initializer/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: *'
_class
loc:@training/Adam/beta_1
?
training/Adam/beta_1VarHandleOp*
shape: *
_output_shapes
: *%
shared_nametraining/Adam/beta_1*'
_class
loc:@training/Adam/beta_1*
dtype0
y
5training/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
?
training/Adam/beta_1/AssignAssignVariableOptraining/Adam/beta_1.training/Adam/beta_1/Initializer/initial_value*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
?
.training/Adam/beta_2/Initializer/initial_valueConst*
_output_shapes
: *'
_class
loc:@training/Adam/beta_2*
dtype0*
valueB
 *w??
?
training/Adam/beta_2VarHandleOp*
dtype0*
_output_shapes
: *'
_class
loc:@training/Adam/beta_2*%
shared_nametraining/Adam/beta_2*
shape: 
y
5training/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
?
training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
?
-training/Adam/decay/Initializer/initial_valueConst*
dtype0*&
_class
loc:@training/Adam/decay*
valueB
 *    *
_output_shapes
: 
?
training/Adam/decayVarHandleOp*
shape: *
dtype0*$
shared_nametraining/Adam/decay*
_output_shapes
: *&
_class
loc:@training/Adam/decay
w
4training/Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/decay*
_output_shapes
: 

training/Adam/decay/AssignAssignVariableOptraining/Adam/decay-training/Adam/decay/Initializer/initial_value*
dtype0
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0
?
5training/Adam/learning_rate/Initializer/initial_valueConst*
_output_shapes
: *
valueB
 *o?:*.
_class$
" loc:@training/Adam/learning_rate*
dtype0
?
training/Adam/learning_rateVarHandleOp*
_output_shapes
: *,
shared_nametraining/Adam/learning_rate*.
_class$
" loc:@training/Adam/learning_rate*
shape: *
dtype0
?
<training/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
?
"training/Adam/learning_rate/AssignAssignVariableOptraining/Adam/learning_rate5training/Adam/learning_rate/Initializer/initial_value*
dtype0
?
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
?
@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*!
_class
loc:@dense_1/kernel*
valueB"?      
?
6training/Adam/dense_1/kernel/m/Initializer/zeros/ConstConst*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0*
valueB
 *    
?
0training/Adam/dense_1/kernel/m/Initializer/zerosFill@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/m/Initializer/zeros/Const*
T0* 
_output_shapes
:
??*!
_class
loc:@dense_1/kernel
?
training/Adam/dense_1/kernel/mVarHandleOp*/
shared_name training/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
shape:
??*
dtype0
?
?training/Adam/dense_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
?
%training/Adam/dense_1/kernel/m/AssignAssignVariableOptraining/Adam/dense_1/kernel/m0training/Adam/dense_1/kernel/m/Initializer/zeros*
dtype0
?
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*
dtype0* 
_output_shapes
:
??*!
_class
loc:@dense_1/kernel
?
>training/Adam/dense_1/bias/m/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
_class
loc:@dense_1/bias*
valueB:?
?
4training/Adam/dense_1/bias/m/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_class
loc:@dense_1/bias*
_output_shapes
: 
?
.training/Adam/dense_1/bias/m/Initializer/zerosFill>training/Adam/dense_1/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_1/bias/m/Initializer/zeros/Const*
_output_shapes	
:?*
T0*
_class
loc:@dense_1/bias
?
training/Adam/dense_1/bias/mVarHandleOp*
_class
loc:@dense_1/bias*
shape:?*
_output_shapes
: *
dtype0*-
shared_nametraining/Adam/dense_1/bias/m
?
=training/Adam/dense_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
_output_shapes
: 
?
#training/Adam/dense_1/bias/m/AssignAssignVariableOptraining/Adam/dense_1/bias/m.training/Adam/dense_1/bias/m/Initializer/zeros*
dtype0
?
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
_output_shapes	
:?*
dtype0
?
@training/Adam/dense_2/kernel/m/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_2/kernel
?
6training/Adam/dense_2/kernel/m/Initializer/zeros/ConstConst*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: *
valueB
 *    
?
0training/Adam/dense_2/kernel/m/Initializer/zerosFill@training/Adam/dense_2/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_2/kernel/m/Initializer/zeros/Const* 
_output_shapes
:
??*
T0*!
_class
loc:@dense_2/kernel
?
training/Adam/dense_2/kernel/mVarHandleOp*
shape:
??*/
shared_name training/Adam/dense_2/kernel/m*
_output_shapes
: *!
_class
loc:@dense_2/kernel*
dtype0
?
?training/Adam/dense_2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/kernel/m*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
?
%training/Adam/dense_2/kernel/m/AssignAssignVariableOptraining/Adam/dense_2/kernel/m0training/Adam/dense_2/kernel/m/Initializer/zeros*
dtype0
?
2training/Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/m*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
??*
dtype0
?
>training/Adam/dense_2/bias/m/Initializer/zeros/shape_as_tensorConst*
valueB:?*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
:
?
4training/Adam/dense_2/bias/m/Initializer/zeros/ConstConst*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
: *
valueB
 *    
?
.training/Adam/dense_2/bias/m/Initializer/zerosFill>training/Adam/dense_2/bias/m/Initializer/zeros/shape_as_tensor4training/Adam/dense_2/bias/m/Initializer/zeros/Const*
_output_shapes	
:?*
T0*
_class
loc:@dense_2/bias
?
training/Adam/dense_2/bias/mVarHandleOp*-
shared_nametraining/Adam/dense_2/bias/m*
_output_shapes
: *
dtype0*
shape:?*
_class
loc:@dense_2/bias
?
=training/Adam/dense_2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/bias/m*
_class
loc:@dense_2/bias*
_output_shapes
: 
?
#training/Adam/dense_2/bias/m/AssignAssignVariableOptraining/Adam/dense_2/bias/m.training/Adam/dense_2/bias/m/Initializer/zeros*
dtype0
?
0training/Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/m*
_class
loc:@dense_2/bias*
_output_shapes	
:?*
dtype0
?
@training/Adam/dense_3/kernel/m/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
:*
valueB"   (   
?
6training/Adam/dense_3/kernel/m/Initializer/zeros/ConstConst*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0*
valueB
 *    
?
0training/Adam/dense_3/kernel/m/Initializer/zerosFill@training/Adam/dense_3/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_3/kernel/m/Initializer/zeros/Const*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	?(
?
training/Adam/dense_3/kernel/mVarHandleOp*
shape:	?(*/
shared_name training/Adam/dense_3/kernel/m*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
dtype0
?
?training/Adam/dense_3/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/kernel/m*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
?
%training/Adam/dense_3/kernel/m/AssignAssignVariableOptraining/Adam/dense_3/kernel/m0training/Adam/dense_3/kernel/m/Initializer/zeros*
dtype0
?
2training/Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/m*!
_class
loc:@dense_3/kernel*
_output_shapes
:	?(*
dtype0
?
.training/Adam/dense_3/bias/m/Initializer/zerosConst*
valueB(*    *
_output_shapes
:(*
dtype0*
_class
loc:@dense_3/bias
?
training/Adam/dense_3/bias/mVarHandleOp*
dtype0*-
shared_nametraining/Adam/dense_3/bias/m*
_class
loc:@dense_3/bias*
shape:(*
_output_shapes
: 
?
=training/Adam/dense_3/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/bias/m*
_class
loc:@dense_3/bias*
_output_shapes
: 
?
#training/Adam/dense_3/bias/m/AssignAssignVariableOptraining/Adam/dense_3/bias/m.training/Adam/dense_3/bias/m/Initializer/zeros*
dtype0
?
0training/Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/m*
dtype0*
_class
loc:@dense_3/bias*
_output_shapes
:(
?
@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"?      
?
6training/Adam/dense_1/kernel/v/Initializer/zeros/ConstConst*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0
?
0training/Adam/dense_1/kernel/v/Initializer/zerosFill@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/v/Initializer/zeros/Const*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
??*
T0
?
training/Adam/dense_1/kernel/vVarHandleOp*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
shape:
??*/
shared_name training/Adam/dense_1/kernel/v
?
?training/Adam/dense_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/v*
_output_shapes
: *!
_class
loc:@dense_1/kernel
?
%training/Adam/dense_1/kernel/v/AssignAssignVariableOptraining/Adam/dense_1/kernel/v0training/Adam/dense_1/kernel/v/Initializer/zeros*
dtype0
?
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
dtype0* 
_output_shapes
:
??
?
>training/Adam/dense_1/bias/v/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:?*
dtype0*
_class
loc:@dense_1/bias
?
4training/Adam/dense_1/bias/v/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0*
_class
loc:@dense_1/bias
?
.training/Adam/dense_1/bias/v/Initializer/zerosFill>training/Adam/dense_1/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_1/bias/v/Initializer/zeros/Const*
_output_shapes	
:?*
T0*
_class
loc:@dense_1/bias
?
training/Adam/dense_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_1/bias/v*
shape:?*
_class
loc:@dense_1/bias
?
=training/Adam/dense_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
_output_shapes
: 
?
#training/Adam/dense_1/bias/v/AssignAssignVariableOptraining/Adam/dense_1/bias/v.training/Adam/dense_1/bias/v/Initializer/zeros*
dtype0
?
0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
_output_shapes	
:?*
dtype0*
_class
loc:@dense_1/bias
?
@training/Adam/dense_2/kernel/v/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
dtype0*!
_class
loc:@dense_2/kernel
?
6training/Adam/dense_2/kernel/v/Initializer/zeros/ConstConst*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
0training/Adam/dense_2/kernel/v/Initializer/zerosFill@training/Adam/dense_2/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_2/kernel/v/Initializer/zeros/Const*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
??
?
training/Adam/dense_2/kernel/vVarHandleOp*
dtype0*!
_class
loc:@dense_2/kernel*
shape:
??*/
shared_name training/Adam/dense_2/kernel/v*
_output_shapes
: 
?
?training/Adam/dense_2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/kernel/v*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
?
%training/Adam/dense_2/kernel/v/AssignAssignVariableOptraining/Adam/dense_2/kernel/v0training/Adam/dense_2/kernel/v/Initializer/zeros*
dtype0
?
2training/Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/v*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
??*
dtype0
?
>training/Adam/dense_2/bias/v/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
_class
loc:@dense_2/bias*
valueB:?*
dtype0
?
4training/Adam/dense_2/bias/v/Initializer/zeros/ConstConst*
_class
loc:@dense_2/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
.training/Adam/dense_2/bias/v/Initializer/zerosFill>training/Adam/dense_2/bias/v/Initializer/zeros/shape_as_tensor4training/Adam/dense_2/bias/v/Initializer/zeros/Const*
_class
loc:@dense_2/bias*
T0*
_output_shapes	
:?
?
training/Adam/dense_2/bias/vVarHandleOp*
dtype0*
_class
loc:@dense_2/bias*
shape:?*
_output_shapes
: *-
shared_nametraining/Adam/dense_2/bias/v
?
=training/Adam/dense_2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_2/bias/v*
_output_shapes
: *
_class
loc:@dense_2/bias
?
#training/Adam/dense_2/bias/v/AssignAssignVariableOptraining/Adam/dense_2/bias/v.training/Adam/dense_2/bias/v/Initializer/zeros*
dtype0
?
0training/Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/v*
dtype0*
_output_shapes	
:?*
_class
loc:@dense_2/bias
?
@training/Adam/dense_3/kernel/v/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"   (   *!
_class
loc:@dense_3/kernel*
dtype0
?
6training/Adam/dense_3/kernel/v/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
?
0training/Adam/dense_3/kernel/v/Initializer/zerosFill@training/Adam/dense_3/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_3/kernel/v/Initializer/zeros/Const*
_output_shapes
:	?(*
T0*!
_class
loc:@dense_3/kernel
?
training/Adam/dense_3/kernel/vVarHandleOp*
dtype0*/
shared_name training/Adam/dense_3/kernel/v*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
shape:	?(
?
?training/Adam/dense_3/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/kernel/v*
_output_shapes
: *!
_class
loc:@dense_3/kernel
?
%training/Adam/dense_3/kernel/v/AssignAssignVariableOptraining/Adam/dense_3/kernel/v0training/Adam/dense_3/kernel/v/Initializer/zeros*
dtype0
?
2training/Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/v*
_output_shapes
:	?(*!
_class
loc:@dense_3/kernel*
dtype0
?
.training/Adam/dense_3/bias/v/Initializer/zerosConst*
valueB(*    *
_class
loc:@dense_3/bias*
_output_shapes
:(*
dtype0
?
training/Adam/dense_3/bias/vVarHandleOp*
shape:(*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: *-
shared_nametraining/Adam/dense_3/bias/v
?
=training/Adam/dense_3/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_3/bias/v*
_class
loc:@dense_3/bias*
_output_shapes
: 
?
#training/Adam/dense_3/bias/v/AssignAssignVariableOptraining/Adam/dense_3/bias/v.training/Adam/dense_3/bias/v/Initializer/zeros*
dtype0
?
0training/Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/v*
_output_shapes
:(*
dtype0*
_class
loc:@dense_3/bias
y
%training/Adam/Identity/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
j
training/Adam/IdentityIdentity%training/Adam/Identity/ReadVariableOp*
T0*
_output_shapes
: 
Y
training/Adam/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
U
training/Adam/add/yConst*
dtype0	*
_output_shapes
: *
value	B	 R
n
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
T0	*
_output_shapes
: 
]
training/Adam/CastCasttraining/Adam/add*

DstT0*
_output_shapes
: *

SrcT0	
t
'training/Adam/Identity_1/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_1Identity'training/Adam/Identity_1/ReadVariableOp*
_output_shapes
: *
T0
t
'training/Adam/Identity_2/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_2Identity'training/Adam/Identity_2/ReadVariableOp*
_output_shapes
: *
T0
g
training/Adam/PowPowtraining/Adam/Identity_1training/Adam/Cast*
T0*
_output_shapes
: 
i
training/Adam/Pow_1Powtraining/Adam/Identity_2training/Adam/Cast*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
c
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
N
training/Adam/SqrtSqrttraining/Adam/sub*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
e
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
h
training/Adam/mulMultraining/Adam/Identitytraining/Adam/truediv*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
dtype0*
valueB
 *???3*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
l
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/Identity_1*
_output_shapes
: *
T0
Z
training/Adam/sub_3/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
_output_shapes
: *
T0
?
:training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdamResourceApplyAdamdense_1/kerneltraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
use_locking(*!
_class
loc:@dense_1/kernel
?
8training/Adam/Adam/update_dense_1/bias/ResourceApplyAdamResourceApplyAdamdense_1/biastraining/Adam/dense_1/bias/mtraining/Adam/dense_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
use_locking(*
_class
loc:@dense_1/bias
?
:training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdamResourceApplyAdamdense_2/kerneltraining/Adam/dense_2/kernel/mtraining/Adam/dense_2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_2/MatMul_grad/MatMul_1*!
_class
loc:@dense_2/kernel*
T0*
use_locking(
?
8training/Adam/Adam/update_dense_2/bias/ResourceApplyAdamResourceApplyAdamdense_2/biastraining/Adam/dense_2/bias/mtraining/Adam/dense_2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense_2/bias
?
:training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdamResourceApplyAdamdense_3/kerneltraining/Adam/dense_3/kernel/mtraining/Adam/dense_3/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_3/MatMul_grad/MatMul_1*
use_locking(*
T0*!
_class
loc:@dense_3/kernel
?
8training/Adam/Adam/update_dense_3/bias/ResourceApplyAdamResourceApplyAdamdense_3/biastraining/Adam/dense_3/bias/mtraining/Adam/dense_3/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_class
loc:@dense_3/bias*
use_locking(*
T0
?
training/Adam/Adam/ConstConst9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_2/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_3/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdam*
value	B	 R*
_output_shapes
: *
dtype0	
j
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOpitertraining/Adam/Adam/Const*
dtype0	
?
!training/Adam/Adam/ReadVariableOpReadVariableOpiter'^training/Adam/Adam/AssignAddVariableOp9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_2/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_2/kernel/ResourceApplyAdam9^training/Adam/Adam/update_dense_3/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_3/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp
Z
ConstConst"/device:CPU:0*
valueB Bmodel*
_output_shapes
: *
dtype0
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*g
value^B\BRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
r
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0
?
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
[
AssignVariableOpAssignVariableOptraining/Adam/dense_1/kernel/mIdentity*
dtype0
?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*g
value^B\BRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
RestoreV2_1	RestoreV2ConstRestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_1IdentityRestoreV2_1*
_output_shapes
:*
T0
_
AssignVariableOp_1AssignVariableOptraining/Adam/dense_1/kernel/v
Identity_1*
dtype0
?
RestoreV2_2/tensor_namesConst"/device:CPU:0*e
value\BZBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_2/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:
?
RestoreV2_2	RestoreV2ConstRestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_2IdentityRestoreV2_2*
_output_shapes
:*
T0
]
AssignVariableOp_2AssignVariableOptraining/Adam/dense_1/bias/m
Identity_2*
dtype0
?
RestoreV2_3/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*e
value\BZBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
t
RestoreV2_3/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 
?
RestoreV2_3	RestoreV2ConstRestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_3IdentityRestoreV2_3*
_output_shapes
:*
T0
]
AssignVariableOp_3AssignVariableOptraining/Adam/dense_1/bias/v
Identity_3*
dtype0
?
RestoreV2_4/tensor_namesConst"/device:CPU:0*g
value^B\BRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0
t
RestoreV2_4/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
RestoreV2_4	RestoreV2ConstRestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_4IdentityRestoreV2_4*
_output_shapes
:*
T0
_
AssignVariableOp_4AssignVariableOptraining/Adam/dense_2/kernel/m
Identity_4*
dtype0
?
RestoreV2_5/tensor_namesConst"/device:CPU:0*g
value^B\BRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_5/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
RestoreV2_5	RestoreV2ConstRestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
_
AssignVariableOp_5AssignVariableOptraining/Adam/dense_2/kernel/v
Identity_5*
dtype0
?
RestoreV2_6/tensor_namesConst"/device:CPU:0*
_output_shapes
:*e
value\BZBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
t
RestoreV2_6/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:
?
RestoreV2_6	RestoreV2ConstRestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_6IdentityRestoreV2_6*
_output_shapes
:*
T0
]
AssignVariableOp_6AssignVariableOptraining/Adam/dense_2/bias/m
Identity_6*
dtype0
?
RestoreV2_7/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*e
value\BZBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
t
RestoreV2_7/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 
?
RestoreV2_7	RestoreV2ConstRestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_7IdentityRestoreV2_7*
_output_shapes
:*
T0
]
AssignVariableOp_7AssignVariableOptraining/Adam/dense_2/bias/v
Identity_7*
dtype0
?
RestoreV2_8/tensor_namesConst"/device:CPU:0*
_output_shapes
:*g
value^B\BRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
t
RestoreV2_8/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0
?
RestoreV2_8	RestoreV2ConstRestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_8IdentityRestoreV2_8*
T0*
_output_shapes
:
_
AssignVariableOp_8AssignVariableOptraining/Adam/dense_3/kernel/m
Identity_8*
dtype0
?
RestoreV2_9/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*g
value^B\BRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
t
RestoreV2_9/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:
?
RestoreV2_9	RestoreV2ConstRestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_9IdentityRestoreV2_9*
_output_shapes
:*
T0
_
AssignVariableOp_9AssignVariableOptraining/Adam/dense_3/kernel/v
Identity_9*
dtype0
?
RestoreV2_10/tensor_namesConst"/device:CPU:0*e
value\BZBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0
u
RestoreV2_10/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
?
RestoreV2_10	RestoreV2ConstRestoreV2_10/tensor_namesRestoreV2_10/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
H
Identity_10IdentityRestoreV2_10*
T0*
_output_shapes
:
_
AssignVariableOp_10AssignVariableOptraining/Adam/dense_3/bias/mIdentity_10*
dtype0
?
RestoreV2_11/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*e
value\BZBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
u
RestoreV2_11/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0
?
RestoreV2_11	RestoreV2ConstRestoreV2_11/tensor_namesRestoreV2_11/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_11IdentityRestoreV2_11*
T0*
_output_shapes
:
_
AssignVariableOp_11AssignVariableOptraining/Adam/dense_3/bias/vIdentity_11*
dtype0
?
RestoreV2_12/tensor_namesConst"/device:CPU:0*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
?
RestoreV2_12/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*)
value BB B B B B B B B B B B *
dtype0
?
RestoreV2_12	RestoreV2ConstRestoreV2_12/tensor_namesRestoreV2_12/shape_and_slices"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2	
H
Identity_12IdentityRestoreV2_12*
T0*
_output_shapes
:
O
AssignVariableOp_12AssignVariableOpdense_1/biasIdentity_12*
dtype0
J
Identity_13IdentityRestoreV2_12:1*
T0*
_output_shapes
:
Q
AssignVariableOp_13AssignVariableOpdense_1/kernelIdentity_13*
dtype0
J
Identity_14IdentityRestoreV2_12:2*
_output_shapes
:*
T0
O
AssignVariableOp_14AssignVariableOpdense_2/biasIdentity_14*
dtype0
J
Identity_15IdentityRestoreV2_12:3*
_output_shapes
:*
T0
Q
AssignVariableOp_15AssignVariableOpdense_2/kernelIdentity_15*
dtype0
J
Identity_16IdentityRestoreV2_12:4*
_output_shapes
:*
T0
O
AssignVariableOp_16AssignVariableOpdense_3/biasIdentity_16*
dtype0
J
Identity_17IdentityRestoreV2_12:5*
T0*
_output_shapes
:
Q
AssignVariableOp_17AssignVariableOpdense_3/kernelIdentity_17*
dtype0
J
Identity_18IdentityRestoreV2_12:6*
_output_shapes
:*
T0
W
AssignVariableOp_18AssignVariableOptraining/Adam/beta_1Identity_18*
dtype0
J
Identity_19IdentityRestoreV2_12:7*
_output_shapes
:*
T0
W
AssignVariableOp_19AssignVariableOptraining/Adam/beta_2Identity_19*
dtype0
J
Identity_20IdentityRestoreV2_12:8*
T0*
_output_shapes
:
V
AssignVariableOp_20AssignVariableOptraining/Adam/decayIdentity_20*
dtype0
J
Identity_21IdentityRestoreV2_12:9*
T0	*
_output_shapes
:
G
AssignVariableOp_21AssignVariableOpiterIdentity_21*
dtype0	
K
Identity_22IdentityRestoreV2_12:10*
T0*
_output_shapes
:
^
AssignVariableOp_22AssignVariableOptraining/Adam/learning_rateIdentity_22*
dtype0
G
VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
W
VarIsInitializedOp_1VarIsInitializedOptraining/Adam/decay*
_output_shapes
: 
H
VarIsInitializedOp_2VarIsInitializedOpiter*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
VarIsInitializedOp_4VarIsInitializedOptraining/Adam/dense_1/kernel/m*
_output_shapes
: 
`
VarIsInitializedOp_5VarIsInitializedOptraining/Adam/dense_2/bias/m*
_output_shapes
: 
b
VarIsInitializedOp_6VarIsInitializedOptraining/Adam/dense_3/kernel/m*
_output_shapes
: 
`
VarIsInitializedOp_7VarIsInitializedOptraining/Adam/dense_1/bias/v*
_output_shapes
: 
R
VarIsInitializedOp_8VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
`
VarIsInitializedOp_9VarIsInitializedOptraining/Adam/dense_1/bias/m*
_output_shapes
: 
c
VarIsInitializedOp_10VarIsInitializedOptraining/Adam/dense_2/kernel/m*
_output_shapes
: 
c
VarIsInitializedOp_11VarIsInitializedOptraining/Adam/dense_1/kernel/v*
_output_shapes
: 
Q
VarIsInitializedOp_12VarIsInitializedOpdense_1/bias*
_output_shapes
: 
J
VarIsInitializedOp_13VarIsInitializedOptotal*
_output_shapes
: 
Y
VarIsInitializedOp_14VarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
a
VarIsInitializedOp_15VarIsInitializedOptraining/Adam/dense_3/bias/m*
_output_shapes
: 
a
VarIsInitializedOp_16VarIsInitializedOptraining/Adam/dense_2/bias/v*
_output_shapes
: 
a
VarIsInitializedOp_17VarIsInitializedOptraining/Adam/dense_3/bias/v*
_output_shapes
: 
S
VarIsInitializedOp_18VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_19VarIsInitializedOpdense_2/bias*
_output_shapes
: 
c
VarIsInitializedOp_20VarIsInitializedOptraining/Adam/dense_2/kernel/v*
_output_shapes
: 
c
VarIsInitializedOp_21VarIsInitializedOptraining/Adam/dense_3/kernel/v*
_output_shapes
: 
S
VarIsInitializedOp_22VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
`
VarIsInitializedOp_23VarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
Y
VarIsInitializedOp_24VarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
?
initNoOp^count/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^iter/Assign^total/Assign^training/Adam/beta_1/Assign^training/Adam/beta_2/Assign^training/Adam/decay/Assign$^training/Adam/dense_1/bias/m/Assign$^training/Adam/dense_1/bias/v/Assign&^training/Adam/dense_1/kernel/m/Assign&^training/Adam/dense_1/kernel/v/Assign$^training/Adam/dense_2/bias/m/Assign$^training/Adam/dense_2/bias/v/Assign&^training/Adam/dense_2/kernel/m/Assign&^training/Adam/dense_2/kernel/v/Assign$^training/Adam/dense_3/bias/m/Assign$^training/Adam/dense_3/bias/v/Assign&^training/Adam/dense_3/kernel/m/Assign&^training/Adam/dense_3/kernel/v/Assign#^training/Adam/learning_rate/Assign
W
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B 
W
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B 
?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_315f3cc9fdb04b86b9646314ed5b93fb/part
f

StringJoin
StringJoinConst_2StringJoin/inputs_1"/device:CPU:0*
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
?
SaveV2/shape_and_slicesConst"/device:CPU:0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
?	
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpiter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp2training/Adam/dense_2/kernel/m/Read/ReadVariableOp0training/Adam/dense_2/bias/m/Read/ReadVariableOp2training/Adam/dense_3/kernel/m/Read/ReadVariableOp0training/Adam/dense_3/bias/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp2training/Adam/dense_2/kernel/v/Read/ReadVariableOp0training/Adam/dense_2/bias/v/Read/ReadVariableOp2training/Adam/dense_3/kernel/v/Read/ReadVariableOp0training/Adam/dense_3/bias/v/Read/ReadVariableOp"/device:CPU:0*%
dtypes
2	
h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
|
ShardedFilename_1ShardedFilename
StringJoinShardedFilename_1/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0
q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 
?
SaveV2_1SaveV2ShardedFilename_1SaveV2_1/tensor_namesSaveV2_1/shape_and_slicesConst_1"/device:CPU:0*
dtypes
2
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilenameShardedFilename_1^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
h
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixesConst_2"/device:CPU:0
e
Identity_23IdentityConst_2^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
W
div_no_nan/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
Y
div_no_nan/ReadVariableOp_1ReadVariableOpcount*
_output_shapes
: *
dtype0
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
D
Identity_24Identity
div_no_nan*
_output_shapes
: *
T0
x
metric_op_wrapperConst"^metrics/acc/AssignAddVariableOp_1*
_output_shapes
: *
valueB *
dtype0
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
_output_shapes
: *
dtype0
?
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
?
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices dense_1/bias/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp0training/Adam/dense_2/bias/m/Read/ReadVariableOp0training/Adam/dense_2/bias/v/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp2training/Adam/dense_2/kernel/m/Read/ReadVariableOp2training/Adam/dense_2/kernel/v/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp0training/Adam/dense_3/bias/m/Read/ReadVariableOp0training/Adam/dense_3/bias/v/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp2training/Adam/dense_3/kernel/m/Read/ReadVariableOp2training/Adam/dense_3/kernel/v/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOpiter/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp*%
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_output_shapes
: *
_class
loc:@save/Const
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
S
save/AssignVariableOpAssignVariableOpdense_1/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
_output_shapes
:*
T0
g
save/AssignVariableOp_1AssignVariableOptraining/Adam/dense_1/bias/msave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
g
save/AssignVariableOp_2AssignVariableOptraining/Adam/dense_1/bias/vsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_3AssignVariableOpdense_1/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0*
_output_shapes
:
i
save/AssignVariableOp_4AssignVariableOptraining/Adam/dense_1/kernel/msave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
T0*
_output_shapes
:
i
save/AssignVariableOp_5AssignVariableOptraining/Adam/dense_1/kernel/vsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:6*
_output_shapes
:*
T0
W
save/AssignVariableOp_6AssignVariableOpdense_2/biassave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:7*
_output_shapes
:*
T0
g
save/AssignVariableOp_7AssignVariableOptraining/Adam/dense_2/bias/msave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:8*
_output_shapes
:*
T0
g
save/AssignVariableOp_8AssignVariableOptraining/Adam/dense_2/bias/vsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:9*
T0*
_output_shapes
:
Y
save/AssignVariableOp_9AssignVariableOpdense_2/kernelsave/Identity_9*
dtype0
R
save/Identity_10Identitysave/RestoreV2:10*
_output_shapes
:*
T0
k
save/AssignVariableOp_10AssignVariableOptraining/Adam/dense_2/kernel/msave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:11*
T0*
_output_shapes
:
k
save/AssignVariableOp_11AssignVariableOptraining/Adam/dense_2/kernel/vsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:12*
T0*
_output_shapes
:
Y
save/AssignVariableOp_12AssignVariableOpdense_3/biassave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:13*
_output_shapes
:*
T0
i
save/AssignVariableOp_13AssignVariableOptraining/Adam/dense_3/bias/msave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:14*
_output_shapes
:*
T0
i
save/AssignVariableOp_14AssignVariableOptraining/Adam/dense_3/bias/vsave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:15*
T0*
_output_shapes
:
[
save/AssignVariableOp_15AssignVariableOpdense_3/kernelsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:16*
T0*
_output_shapes
:
k
save/AssignVariableOp_16AssignVariableOptraining/Adam/dense_3/kernel/msave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:17*
_output_shapes
:*
T0
k
save/AssignVariableOp_17AssignVariableOptraining/Adam/dense_3/kernel/vsave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:18*
_output_shapes
:*
T0
a
save/AssignVariableOp_18AssignVariableOptraining/Adam/beta_1save/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:19*
T0*
_output_shapes
:
a
save/AssignVariableOp_19AssignVariableOptraining/Adam/beta_2save/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:20*
T0*
_output_shapes
:
`
save/AssignVariableOp_20AssignVariableOptraining/Adam/decaysave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:21*
_output_shapes
:*
T0	
Q
save/AssignVariableOp_21AssignVariableOpitersave/Identity_21*
dtype0	
R
save/Identity_22Identitysave/RestoreV2:22*
_output_shapes
:*
T0
h
save/AssignVariableOp_22AssignVariableOptraining/Adam/learning_ratesave/Identity_22*
dtype0
?
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
,
init_1NoOp^count/Assign^total/Assign"?D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"?
trainable_variables??
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
?
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08"?
local_variables??
U
total:0total/Assigntotal/Read/ReadVariableOp:0(2total/Initializer/zeros:0@H
U
count:0count/Assigncount/Read/ReadVariableOp:0(2count/Initializer/zeros:0@H"b
global_stepSQ
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H"?
	variables??
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
?
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H
?
training/Adam/beta_1:0training/Adam/beta_1/Assign*training/Adam/beta_1/Read/ReadVariableOp:0(20training/Adam/beta_1/Initializer/initial_value:0H
?
training/Adam/beta_2:0training/Adam/beta_2/Assign*training/Adam/beta_2/Read/ReadVariableOp:0(20training/Adam/beta_2/Initializer/initial_value:0H
?
training/Adam/decay:0training/Adam/decay/Assign)training/Adam/decay/Read/ReadVariableOp:0(2/training/Adam/decay/Initializer/initial_value:0H
?
training/Adam/learning_rate:0"training/Adam/learning_rate/Assign1training/Adam/learning_rate/Read/ReadVariableOp:0(27training/Adam/learning_rate/Initializer/initial_value:0H
?
 training/Adam/dense_1/kernel/m:0%training/Adam/dense_1/kernel/m/Assign4training/Adam/dense_1/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_1/kernel/m/Initializer/zeros:0
?
training/Adam/dense_1/bias/m:0#training/Adam/dense_1/bias/m/Assign2training/Adam/dense_1/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_1/bias/m/Initializer/zeros:0
?
 training/Adam/dense_2/kernel/m:0%training/Adam/dense_2/kernel/m/Assign4training/Adam/dense_2/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_2/kernel/m/Initializer/zeros:0
?
training/Adam/dense_2/bias/m:0#training/Adam/dense_2/bias/m/Assign2training/Adam/dense_2/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_2/bias/m/Initializer/zeros:0
?
 training/Adam/dense_3/kernel/m:0%training/Adam/dense_3/kernel/m/Assign4training/Adam/dense_3/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_3/kernel/m/Initializer/zeros:0
?
training/Adam/dense_3/bias/m:0#training/Adam/dense_3/bias/m/Assign2training/Adam/dense_3/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_3/bias/m/Initializer/zeros:0
?
 training/Adam/dense_1/kernel/v:0%training/Adam/dense_1/kernel/v/Assign4training/Adam/dense_1/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_1/kernel/v/Initializer/zeros:0
?
training/Adam/dense_1/bias/v:0#training/Adam/dense_1/bias/v/Assign2training/Adam/dense_1/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_1/bias/v/Initializer/zeros:0
?
 training/Adam/dense_2/kernel/v:0%training/Adam/dense_2/kernel/v/Assign4training/Adam/dense_2/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_2/kernel/v/Initializer/zeros:0
?
training/Adam/dense_2/bias/v:0#training/Adam/dense_2/bias/v/Assign2training/Adam/dense_2/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_2/bias/v/Initializer/zeros:0
?
 training/Adam/dense_3/kernel/v:0%training/Adam/dense_3/kernel/v/Assign4training/Adam/dense_3/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_3/kernel/v/Initializer/zeros:0
?
training/Adam/dense_3/bias/v:0#training/Adam/dense_3/bias/v/Assign2training/Adam/dense_3/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_3/bias/v/Initializer/zeros:0*Q
__saved_model_train_op75
__saved_model_train_op
training_1/group_deps*@
__saved_model_init_op'%
__saved_model_init_op
init_1*?
train?
8
dense_1_input'
dense_1_input:0??????????
B
dense_3_target0
dense_3_target:0???????????????????
predictions/dense_3(
dense_3/Softmax:0?????????(4
metrics/acc/update_op
metric_op_wrapper:0 (
metrics/acc/value
Identity_24:0 
loss

loss/mul:0 tensorflow/supervised/training??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetype?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(?
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?"eval*1.15.02v1.15.0-rc3-22-g590d6ee8??
r
dense_1_inputPlaceholder*
shape:??????????*
dtype0*(
_output_shapes
:??????????
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"?      
?
-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB
 *:͓?*
dtype0*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
valueB
 *:͓=
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0* 
_output_shapes
:
??
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
T0
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
??*
T0
?
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
??*!
_class
loc:@dense_1/kernel
?
dense_1/kernelVarHandleOp*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
shape:
??*
shared_namedense_1/kernel*
dtype0
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
?
.dense_1/bias/Initializer/zeros/shape_as_tensorConst*
valueB:?*
_class
loc:@dense_1/bias*
_output_shapes
:*
dtype0
?
$dense_1/bias/Initializer/zeros/ConstConst*
_class
loc:@dense_1/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
dense_1/bias/Initializer/zerosFill.dense_1/bias/Initializer/zeros/shape_as_tensor$dense_1/bias/Initializer/zeros/Const*
_output_shapes	
:?*
T0*
_class
loc:@dense_1/bias
?
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: *
shape:?
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
n
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0* 
_output_shapes
:
??
y
dense_1/MatMulMatMuldense_1_inputdense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????
h
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
}
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*(
_output_shapes
:??????????*
T0
X
dense_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:??????????
_
dropout_1/IdentityIdentitydense_1/Relu*
T0*(
_output_shapes
:??????????
?
/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_2/kernel*
dtype0
?
-dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *׳]?*
dtype0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
?
-dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]=*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
?
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_2/kernel*
dtype0*
T0* 
_output_shapes
:
??
?
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
T0
?
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
??*!
_class
loc:@dense_2/kernel*
T0
?
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min* 
_output_shapes
:
??*!
_class
loc:@dense_2/kernel*
T0
?
dense_2/kernelVarHandleOp*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
dtype0*
shared_namedense_2/kernel*
shape:
??
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
q
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
??
?
.dense_2/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:?*
_output_shapes
:*
_class
loc:@dense_2/bias
?
$dense_2/bias/Initializer/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
_class
loc:@dense_2/bias*
dtype0
?
dense_2/bias/Initializer/zerosFill.dense_2/bias/Initializer/zeros/shape_as_tensor$dense_2/bias/Initializer/zeros/Const*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:?
?
dense_2/biasVarHandleOp*
_class
loc:@dense_2/bias*
shared_namedense_2/bias*
dtype0*
shape:?*
_output_shapes
: 
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
b
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
dtype0
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
n
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
??
~
dense_2/MatMulMatMuldropout_1/Identitydense_2/MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0
h
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes	
:?
}
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:??????????
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:??????????*
T0
_
dropout_2/IdentityIdentitydense_2/Relu*
T0*(
_output_shapes
:??????????
?
/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"   (   *
_output_shapes
:*
dtype0*!
_class
loc:@dense_3/kernel
?
-dense_3/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_3/kernel*
valueB
 *?ʙ?*
_output_shapes
: *
dtype0
?
-dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
valueB
 *?ʙ=
?
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes
:	?(*
dtype0
?
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 
?
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?(*!
_class
loc:@dense_3/kernel*
T0
?
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
:	?(*!
_class
loc:@dense_3/kernel*
T0
?
dense_3/kernelVarHandleOp*!
_class
loc:@dense_3/kernel*
_output_shapes
: *
dtype0*
shared_namedense_3/kernel*
shape:	?(
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
q
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*
dtype0
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?(*
dtype0
?
dense_3/bias/Initializer/zerosConst*
valueB(*    *
_output_shapes
:(*
_class
loc:@dense_3/bias*
dtype0
?
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
_class
loc:@dense_3/bias*
shape:(*
shared_namedense_3/bias
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
dtype0
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
dtype0*
_output_shapes
:(
m
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes
:	?(
}
dense_3/MatMulMatMuldropout_2/Identitydense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(
g
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:(*
dtype0
|
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*'
_output_shapes
:?????????(*
T0
]
dense_3/SoftmaxSoftmaxdense_3/BiasAdd*
T0*'
_output_shapes
:?????????(
?
dense_3_targetPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
v
total/Initializer/zerosConst*
dtype0*
_class

loc:@total*
_output_shapes
: *
valueB
 *    
x
totalVarHandleOp*
_class

loc:@total*
dtype0*
shared_nametotal*
_output_shapes
: *
shape: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
v
count/Initializer/zerosConst*
dtype0*
valueB
 *    *
_output_shapes
: *
_class

loc:@count
x
countVarHandleOp*
dtype0*
shape: *
_class

loc:@count*
_output_shapes
: *
shared_namecount
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
g
metrics/acc/ArgMax/dimensionConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
x
metrics/acc/ArgMaxArgMaxdense_3_targetmetrics/acc/ArgMax/dimension*
T0*#
_output_shapes
:?????????
i
metrics/acc/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
?????????
}
metrics/acc/ArgMax_1ArgMaxdense_3/Softmaxmetrics/acc/ArgMax_1/dimension*#
_output_shapes
:?????????*
T0
r
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
T0	*#
_output_shapes
:?????????
h
metrics/acc/CastCastmetrics/acc/Equal*#
_output_shapes
:?????????*

SrcT0
*

DstT0
[
metrics/acc/ConstConst*
valueB: *
_output_shapes
:*
dtype0
\
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*
T0*
_output_shapes
: 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
?
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
K
metrics/acc/SizeSizemetrics/acc/Cast*
T0*
_output_shapes
: 
\
metrics/acc/Cast_1Castmetrics/acc/Size*

DstT0*

SrcT0*
_output_shapes
: 
?
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_1 ^metrics/acc/AssignAddVariableOp*
dtype0
?
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
?
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
?
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
?
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
\
loss/dense_3_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
z
8loss/dense_3_loss/softmax_cross_entropy_with_logits/RankConst*
dtype0*
_output_shapes
: *
value	B :
x
9loss/dense_3_loss/softmax_cross_entropy_with_logits/ShapeShapedense_3/BiasAdd*
T0*
_output_shapes
:
|
:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
z
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_1Shapedense_3/BiasAdd*
_output_shapes
:*
T0
{
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :
?
7loss/dense_3_loss/softmax_cross_entropy_with_logits/SubSub:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_19loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
?
?loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/beginPack7loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub*
T0*
N*
_output_shapes
:
?
>loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
?
9loss/dense_3_loss/softmax_cross_entropy_with_logits/SliceSlice;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_1?loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/begin>loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
_output_shapes
:*
T0
?
Closs/dense_3_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
dtype0*
valueB:
?????????*
_output_shapes
:
?
?loss/dense_3_loss/softmax_cross_entropy_with_logits/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
:loss/dense_3_loss/softmax_cross_entropy_with_logits/concatConcatV2Closs/dense_3_loss/softmax_cross_entropy_with_logits/concat/values_09loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice?loss/dense_3_loss/softmax_cross_entropy_with_logits/concat/axis*
T0*
_output_shapes
:*
N
?
;loss/dense_3_loss/softmax_cross_entropy_with_logits/ReshapeReshapedense_3/BiasAdd:loss/dense_3_loss/softmax_cross_entropy_with_logits/concat*0
_output_shapes
:??????????????????*
T0
|
:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
y
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_2Shapedense_3_target*
_output_shapes
:*
T0
}
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
?
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1Sub:loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank_2;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
?
Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_1*
_output_shapes
:*
N*
T0
?
@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1Slice;loss/dense_3_loss/softmax_cross_entropy_with_logits/Shape_2Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/begin@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1/size*
_output_shapes
:*
T0*
Index0
?
Eloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
?????????*
_output_shapes
:*
dtype0
?
Aloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
<loss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Eloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/values_0;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_1Aloss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1/axis*
T0*
N*
_output_shapes
:
?
=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_3_target<loss/dense_3_loss/softmax_cross_entropy_with_logits/concat_1*0
_output_shapes
:??????????????????*
T0
?
3loss/dense_3_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits;loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_1*?
_output_shapes-
+:?????????:??????????????????*
T0
}
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :
?
9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2Sub8loss/dense_3_loss/softmax_cross_entropy_with_logits/Rank;loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
?
Aloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack9loss/dense_3_loss/softmax_cross_entropy_with_logits/Sub_2*
N*
T0*
_output_shapes
:
?
;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2Slice9loss/dense_3_loss/softmax_cross_entropy_with_logits/ShapeAloss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/begin@loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2/size*
T0*
Index0*
_output_shapes
:
?
=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape3loss/dense_3_loss/softmax_cross_entropy_with_logits;loss/dense_3_loss/softmax_cross_entropy_with_logits/Slice_2*
T0*#
_output_shapes
:?????????
k
&loss/dense_3_loss/weighted_loss/Cast/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
Tloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
Sloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
_output_shapes
: *
value	B : *
dtype0
?
Sloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
_output_shapes
:
?
Rloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
j
bloss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
?
Aloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_2c^loss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
_output_shapes
:
?
Aloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_3_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
;loss/dense_3_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_3_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:?????????
?
1loss/dense_3_loss/weighted_loss/broadcast_weightsMul&loss/dense_3_loss/weighted_loss/Cast/x;loss/dense_3_loss/weighted_loss/broadcast_weights/ones_like*#
_output_shapes
:?????????*
T0
?
#loss/dense_3_loss/weighted_loss/MulMul=loss/dense_3_loss/softmax_cross_entropy_with_logits/Reshape_21loss/dense_3_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:?????????
c
loss/dense_3_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
}
loss/dense_3_loss/SumSum#loss/dense_3_loss/weighted_loss/Mulloss/dense_3_loss/Const_1*
T0*
_output_shapes
: 
l
loss/dense_3_loss/num_elementsSize#loss/dense_3_loss/weighted_loss/Mul*
_output_shapes
: *
T0
{
#loss/dense_3_loss/num_elements/CastCastloss/dense_3_loss/num_elements*

SrcT0*
_output_shapes
: *

DstT0
\
loss/dense_3_loss/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
q
loss/dense_3_loss/Sum_1Sumloss/dense_3_loss/Sumloss/dense_3_loss/Const_2*
_output_shapes
: *
T0
?
loss/dense_3_loss/valueDivNoNanloss/dense_3_loss/Sum_1#loss/dense_3_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
U
loss/mulMul
loss/mul/xloss/dense_3_loss/value*
T0*
_output_shapes
: 
q
iter/Initializer/zerosConst*
dtype0	*
value	B	 R *
_output_shapes
: *
_class
	loc:@iter
u
iterVarHandleOp*
_class
	loc:@iter*
_output_shapes
: *
dtype0	*
shared_nameiter*
shape: 
Y
%iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpiter*
_output_shapes
: 
J
iter/AssignAssignVariableOpiteriter/Initializer/zeros*
dtype0	
U
iter/Read/ReadVariableOpReadVariableOpiter*
dtype0	*
_output_shapes
: 
(
evaluation/group_depsNoOp	^loss/mul
Z
ConstConst"/device:CPU:0*
_output_shapes
: *
valueB Bmodel*
dtype0
?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0
~
RestoreV2/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
_output_shapes
:*
dtype0
?
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
I
AssignVariableOpAssignVariableOpdense_1/biasIdentity*
dtype0
F

Identity_1IdentityRestoreV2:1*
T0*
_output_shapes
:
O
AssignVariableOp_1AssignVariableOpdense_1/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:2*
T0*
_output_shapes
:
M
AssignVariableOp_2AssignVariableOpdense_2/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:3*
_output_shapes
:*
T0
O
AssignVariableOp_3AssignVariableOpdense_2/kernel
Identity_3*
dtype0
F

Identity_4IdentityRestoreV2:4*
T0*
_output_shapes
:
M
AssignVariableOp_4AssignVariableOpdense_3/bias
Identity_4*
dtype0
F

Identity_5IdentityRestoreV2:5*
T0*
_output_shapes
:
O
AssignVariableOp_5AssignVariableOpdense_3/kernel
Identity_5*
dtype0
F

Identity_6IdentityRestoreV2:6*
_output_shapes
:*
T0	
E
AssignVariableOp_6AssignVariableOpiter
Identity_6*
dtype0	
P
VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
P
VarIsInitializedOp_1VarIsInitializedOpdense_1/bias*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense_3/bias*
_output_shapes
: 
H
VarIsInitializedOp_3VarIsInitializedOpiter*
_output_shapes
: 
P
VarIsInitializedOp_4VarIsInitializedOpdense_2/bias*
_output_shapes
: 
I
VarIsInitializedOp_5VarIsInitializedOpcount*
_output_shapes
: 
R
VarIsInitializedOp_6VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
R
VarIsInitializedOp_7VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
I
VarIsInitializedOp_8VarIsInitializedOptotal*
_output_shapes
: 
?
initNoOp^count/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign^iter/Assign^total/Assign
W
div_no_nan/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
Y
div_no_nan/ReadVariableOp_1ReadVariableOpcount*
_output_shapes
: *
dtype0
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
C

Identity_7Identity
div_no_nan*
_output_shapes
: *
T0
x
metric_op_wrapperConst"^metrics/acc/AssignAddVariableOp_1*
_output_shapes
: *
valueB *
dtype0
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: *
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
?
save/SaveV2/tensor_namesConst*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
q
save/SaveV2/shape_and_slicesConst*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOpiter/Read/ReadVariableOp*
dtypes
	2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
	2	*0
_output_shapes
:::::::
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
S
save/AssignVariableOpAssignVariableOpdense_1/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
_output_shapes
:*
T0
Y
save/AssignVariableOp_1AssignVariableOpdense_1/kernelsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
_output_shapes
:*
T0
W
save/AssignVariableOp_2AssignVariableOpdense_2/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
_output_shapes
:*
T0
Y
save/AssignVariableOp_3AssignVariableOpdense_2/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
_output_shapes
:*
T0
W
save/AssignVariableOp_4AssignVariableOpdense_3/biassave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
_output_shapes
:*
T0
Y
save/AssignVariableOp_5AssignVariableOpdense_3/kernelsave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:6*
T0	*
_output_shapes
:
O
save/AssignVariableOp_6AssignVariableOpitersave/Identity_6*
dtype0	
?
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6
,
init_1NoOp^count/Assign^total/Assign"?D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"?
local_variables??
U
total:0total/Assigntotal/Read/ReadVariableOp:0(2total/Initializer/zeros:0@H
U
count:0count/Assigncount/Read/ReadVariableOp:0(2count/Initializer/zeros:0@H"?
trainable_variables??
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
?
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08"?
	variables??
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
?
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H"b
global_stepSQ
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H*?
eval?
8
dense_1_input'
dense_1_input:0??????????
B
dense_3_target0
dense_3_target:0??????????????????'
metrics/acc/value
Identity_7:0 
loss

loss/mul:0 ?
predictions/dense_3(
dense_3/Softmax:0?????????(4
metrics/acc/update_op
metric_op_wrapper:0 tensorflow/supervised/eval*@
__saved_model_init_op'%
__saved_model_init_op
init_1Ж
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
9
Softmax
logits"T
softmax"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?"serve*1.15.02v1.15.0-rc3-22-g590d6ee8?{
r
dense_1_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"?      *!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0
?
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓?*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0
?
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0
?
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0* 
_output_shapes
:
??*!
_class
loc:@dense_1/kernel*
dtype0
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
: 
?
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
??*
T0
?
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
??*
T0*!
_class
loc:@dense_1/kernel
?
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*
shape:
??*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
?
.dense_1/bias/Initializer/zeros/shape_as_tensorConst*
valueB:?*
dtype0*
_output_shapes
:*
_class
loc:@dense_1/bias
?
$dense_1/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@dense_1/bias
?
dense_1/bias/Initializer/zerosFill.dense_1/bias/Initializer/zeros/shape_as_tensor$dense_1/bias/Initializer/zeros/Const*
_class
loc:@dense_1/bias*
T0*
_output_shapes	
:?
?
dense_1/biasVarHandleOp*
_class
loc:@dense_1/bias*
shape:?*
dtype0*
shared_namedense_1/bias*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
n
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
y
dense_1/MatMulMatMuldense_1_inputdense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????
h
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:?
}
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:??????????
X
dense_1/ReluReludense_1/BiasAdd*(
_output_shapes
:??????????*
T0
_
dropout_1/IdentityIdentitydense_1/Relu*
T0*(
_output_shapes
:??????????
?
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
dtype0*
valueB"      *
_output_shapes
:
?
-dense_2/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_2/kernel*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
?
-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
dtype0*
valueB
 *׳]=*
_output_shapes
: 
?
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
T0*
dtype0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
??
?
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *!
_class
loc:@dense_2/kernel*
T0
?
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
??*
T0
?
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
??
?
dense_2/kernelVarHandleOp*
_output_shapes
: *
shared_namedense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
shape:
??
m
/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/kernel*
_output_shapes
: 
q
dense_2/kernel/AssignAssignVariableOpdense_2/kernel)dense_2/kernel/Initializer/random_uniform*
dtype0
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
??
?
.dense_2/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:?*
dtype0*
_class
loc:@dense_2/bias
?
$dense_2/bias/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_class
loc:@dense_2/bias*
_output_shapes
: 
?
dense_2/bias/Initializer/zerosFill.dense_2/bias/Initializer/zeros/shape_as_tensor$dense_2/bias/Initializer/zeros/Const*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:?
?
dense_2/biasVarHandleOp*
shared_namedense_2/bias*
_output_shapes
: *
dtype0*
_class
loc:@dense_2/bias*
shape:?
i
-dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_2/bias*
_output_shapes
: 
b
dense_2/bias/AssignAssignVariableOpdense_2/biasdense_2/bias/Initializer/zeros*
dtype0
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
n
dense_2/MatMul/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0* 
_output_shapes
:
??
~
dense_2/MatMulMatMuldropout_1/Identitydense_2/MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0
h
dense_2/BiasAdd/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
}
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/BiasAdd/ReadVariableOp*(
_output_shapes
:??????????*
T0
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:??????????*
T0
_
dropout_2/IdentityIdentitydense_2/Relu*
T0*(
_output_shapes
:??????????
?
/dense_3/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_3/kernel*
valueB"   (   *
dtype0*
_output_shapes
:
?
-dense_3/kernel/Initializer/random_uniform/minConst*
valueB
 *?ʙ?*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0
?
-dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *?ʙ=*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
?
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes
:	?(
?
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_3/kernel
?
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes
:	?(
?
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	?(
?
dense_3/kernelVarHandleOp*
shared_namedense_3/kernel*
dtype0*
_output_shapes
: *
shape:	?(*!
_class
loc:@dense_3/kernel
m
/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/kernel*
_output_shapes
: 
q
dense_3/kernel/AssignAssignVariableOpdense_3/kernel)dense_3/kernel/Initializer/random_uniform*
dtype0
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes
:	?(
?
dense_3/bias/Initializer/zerosConst*
_output_shapes
:(*
dtype0*
_class
loc:@dense_3/bias*
valueB(*    
?
dense_3/biasVarHandleOp*
_class
loc:@dense_3/bias*
shared_namedense_3/bias*
_output_shapes
: *
dtype0*
shape:(
i
-dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_3/bias*
_output_shapes
: 
b
dense_3/bias/AssignAssignVariableOpdense_3/biasdense_3/bias/Initializer/zeros*
dtype0
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:(*
dtype0
m
dense_3/MatMul/ReadVariableOpReadVariableOpdense_3/kernel*
dtype0*
_output_shapes
:	?(
}
dense_3/MatMulMatMuldropout_2/Identitydense_3/MatMul/ReadVariableOp*'
_output_shapes
:?????????(*
T0
g
dense_3/BiasAdd/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:(*
dtype0
|
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????(
]
dense_3/SoftmaxSoftmaxdense_3/BiasAdd*
T0*'
_output_shapes
:?????????(
,
predict/group_depsNoOp^dense_3/Softmax
Z
ConstConst"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
|
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB B B B B B 
?
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
I
AssignVariableOpAssignVariableOpdense_1/biasIdentity*
dtype0
F

Identity_1IdentityRestoreV2:1*
_output_shapes
:*
T0
O
AssignVariableOp_1AssignVariableOpdense_1/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:2*
T0*
_output_shapes
:
M
AssignVariableOp_2AssignVariableOpdense_2/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:3*
_output_shapes
:*
T0
O
AssignVariableOp_3AssignVariableOpdense_2/kernel
Identity_3*
dtype0
F

Identity_4IdentityRestoreV2:4*
_output_shapes
:*
T0
M
AssignVariableOp_4AssignVariableOpdense_3/bias
Identity_4*
dtype0
F

Identity_5IdentityRestoreV2:5*
_output_shapes
:*
T0
O
AssignVariableOp_5AssignVariableOpdense_3/kernel
Identity_5*
dtype0
P
VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
P
VarIsInitializedOp_1VarIsInitializedOpdense_2/bias*
_output_shapes
: 
R
VarIsInitializedOp_2VarIsInitializedOpdense_3/kernel*
_output_shapes
: 
R
VarIsInitializedOp_3VarIsInitializedOpdense_2/kernel*
_output_shapes
: 
P
VarIsInitializedOp_4VarIsInitializedOpdense_3/bias*
_output_shapes
: 
P
VarIsInitializedOp_5VarIsInitializedOpdense_1/bias*
_output_shapes
: 
?
initNoOp^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^dense_3/bias/Assign^dense_3/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: *
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0
o
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueBB B B B B B 
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp*
dtypes

2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B B *
_output_shapes
:*
dtype0
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2
L
save/IdentityIdentitysave/RestoreV2*
_output_shapes
:*
T0
S
save/AssignVariableOpAssignVariableOpdense_1/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
Y
save/AssignVariableOp_1AssignVariableOpdense_1/kernelsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
_output_shapes
:*
T0
W
save/AssignVariableOp_2AssignVariableOpdense_2/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
_output_shapes
:*
T0
Y
save/AssignVariableOp_3AssignVariableOpdense_2/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
_output_shapes
:*
T0
W
save/AssignVariableOp_4AssignVariableOpdense_3/biassave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
T0*
_output_shapes
:
Y
save/AssignVariableOp_5AssignVariableOpdense_3/kernelsave/Identity_5*
dtype0
?
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5

init_1NoOp"?D
save/Const:0save/control_dependency:0save/restore_all 5 @F8"?
	variables??
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
?
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08"?
trainable_variables??
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
?
dense_2/kernel:0dense_2/kernel/Assign$dense_2/kernel/Read/ReadVariableOp:0(2+dense_2/kernel/Initializer/random_uniform:08
o
dense_2/bias:0dense_2/bias/Assign"dense_2/bias/Read/ReadVariableOp:0(2 dense_2/bias/Initializer/zeros:08
?
dense_3/kernel:0dense_3/kernel/Assign$dense_3/kernel/Read/ReadVariableOp:0(2+dense_3/kernel/Initializer/random_uniform:08
o
dense_3/bias:0dense_3/bias/Assign"dense_3/bias/Read/ReadVariableOp:0(2 dense_3/bias/Initializer/zeros:08*@
__saved_model_init_op'%
__saved_model_init_op
init_1*?
serving_default?
8
dense_1_input'
dense_1_input:0??????????3
dense_3(
dense_3/Softmax:0?????????(tensorflow/serving/predict