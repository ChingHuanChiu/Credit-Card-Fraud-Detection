??
??
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
)credict_fraud_detect/encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*:
shared_name+)credict_fraud_detect/encoder/dense/kernel
?
=credict_fraud_detect/encoder/dense/kernel/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/encoder/dense/kernel* 
_output_shapes
:
??*
dtype0
?
'credict_fraud_detect/encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'credict_fraud_detect/encoder/dense/bias
?
;credict_fraud_detect/encoder/dense/bias/Read/ReadVariableOpReadVariableOp'credict_fraud_detect/encoder/dense/bias*
_output_shapes	
:?*
dtype0
?
+credict_fraud_detect/encoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*<
shared_name-+credict_fraud_detect/encoder/dense_1/kernel
?
?credict_fraud_detect/encoder/dense_1/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/encoder/dense_1/kernel*
_output_shapes
:	?@*
dtype0
?
)credict_fraud_detect/encoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)credict_fraud_detect/encoder/dense_1/bias
?
=credict_fraud_detect/encoder/dense_1/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/encoder/dense_1/bias*
_output_shapes
:@*
dtype0
?
+credict_fraud_detect/encoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *<
shared_name-+credict_fraud_detect/encoder/dense_2/kernel
?
?credict_fraud_detect/encoder/dense_2/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/encoder/dense_2/kernel*
_output_shapes

:@ *
dtype0
?
)credict_fraud_detect/encoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)credict_fraud_detect/encoder/dense_2/bias
?
=credict_fraud_detect/encoder/dense_2/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/encoder/dense_2/bias*
_output_shapes
: *
dtype0
?
+credict_fraud_detect/encoder/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *<
shared_name-+credict_fraud_detect/encoder/dense_3/kernel
?
?credict_fraud_detect/encoder/dense_3/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/encoder/dense_3/kernel*
_output_shapes

: *
dtype0
?
)credict_fraud_detect/encoder/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)credict_fraud_detect/encoder/dense_3/bias
?
=credict_fraud_detect/encoder/dense_3/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/encoder/dense_3/bias*
_output_shapes
:*
dtype0
?
+credict_fraud_detect/encoder/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+credict_fraud_detect/encoder/dense_4/kernel
?
?credict_fraud_detect/encoder/dense_4/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/encoder/dense_4/kernel*
_output_shapes

:*
dtype0
?
)credict_fraud_detect/encoder/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)credict_fraud_detect/encoder/dense_4/bias
?
=credict_fraud_detect/encoder/dense_4/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/encoder/dense_4/bias*
_output_shapes
:*
dtype0
?
+credict_fraud_detect/decoder/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+credict_fraud_detect/decoder/dense_5/kernel
?
?credict_fraud_detect/decoder/dense_5/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/decoder/dense_5/kernel*
_output_shapes

:*
dtype0
?
)credict_fraud_detect/decoder/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)credict_fraud_detect/decoder/dense_5/bias
?
=credict_fraud_detect/decoder/dense_5/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/decoder/dense_5/bias*
_output_shapes
:*
dtype0
?
+credict_fraud_detect/decoder/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *<
shared_name-+credict_fraud_detect/decoder/dense_6/kernel
?
?credict_fraud_detect/decoder/dense_6/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/decoder/dense_6/kernel*
_output_shapes

: *
dtype0
?
)credict_fraud_detect/decoder/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)credict_fraud_detect/decoder/dense_6/bias
?
=credict_fraud_detect/decoder/dense_6/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/decoder/dense_6/bias*
_output_shapes
: *
dtype0
?
+credict_fraud_detect/decoder/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*<
shared_name-+credict_fraud_detect/decoder/dense_7/kernel
?
?credict_fraud_detect/decoder/dense_7/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/decoder/dense_7/kernel*
_output_shapes

: @*
dtype0
?
)credict_fraud_detect/decoder/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)credict_fraud_detect/decoder/dense_7/bias
?
=credict_fraud_detect/decoder/dense_7/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/decoder/dense_7/bias*
_output_shapes
:@*
dtype0
?
+credict_fraud_detect/decoder/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*<
shared_name-+credict_fraud_detect/decoder/dense_8/kernel
?
?credict_fraud_detect/decoder/dense_8/kernel/Read/ReadVariableOpReadVariableOp+credict_fraud_detect/decoder/dense_8/kernel*
_output_shapes
:	@?*
dtype0
?
)credict_fraud_detect/decoder/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)credict_fraud_detect/decoder/dense_8/bias
?
=credict_fraud_detect/decoder/dense_8/bias/Read/ReadVariableOpReadVariableOp)credict_fraud_detect/decoder/dense_8/bias*
_output_shapes	
:?*
dtype0
?
#credict_fraud_detect/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#credict_fraud_detect/dense_9/kernel
?
7credict_fraud_detect/dense_9/kernel/Read/ReadVariableOpReadVariableOp#credict_fraud_detect/dense_9/kernel*
_output_shapes
:	?*
dtype0
?
!credict_fraud_detect/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!credict_fraud_detect/dense_9/bias
?
5credict_fraud_detect/dense_9/bias/Read/ReadVariableOpReadVariableOp!credict_fraud_detect/dense_9/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
?
0Adam/credict_fraud_detect/encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*A
shared_name20Adam/credict_fraud_detect/encoder/dense/kernel/m
?
DAdam/credict_fraud_detect/encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
.Adam/credict_fraud_detect/encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/credict_fraud_detect/encoder/dense/bias/m
?
BAdam/credict_fraud_detect/encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp.Adam/credict_fraud_detect/encoder/dense/bias/m*
_output_shapes	
:?*
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*C
shared_name42Adam/credict_fraud_detect/encoder/dense_1/kernel/m
?
FAdam/credict_fraud_detect/encoder/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_1/kernel/m*
_output_shapes
:	?@*
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/credict_fraud_detect/encoder/dense_1/bias/m
?
DAdam/credict_fraud_detect/encoder/dense_1/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_1/bias/m*
_output_shapes
:@*
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *C
shared_name42Adam/credict_fraud_detect/encoder/dense_2/kernel/m
?
FAdam/credict_fraud_detect/encoder/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_2/kernel/m*
_output_shapes

:@ *
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/credict_fraud_detect/encoder/dense_2/bias/m
?
DAdam/credict_fraud_detect/encoder/dense_2/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_2/bias/m*
_output_shapes
: *
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42Adam/credict_fraud_detect/encoder/dense_3/kernel/m
?
FAdam/credict_fraud_detect/encoder/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_3/kernel/m*
_output_shapes

: *
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/credict_fraud_detect/encoder/dense_3/bias/m
?
DAdam/credict_fraud_detect/encoder/dense_3/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_3/bias/m*
_output_shapes
:*
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/credict_fraud_detect/encoder/dense_4/kernel/m
?
FAdam/credict_fraud_detect/encoder/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_4/kernel/m*
_output_shapes

:*
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/credict_fraud_detect/encoder/dense_4/bias/m
?
DAdam/credict_fraud_detect/encoder/dense_4/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_4/bias/m*
_output_shapes
:*
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/credict_fraud_detect/decoder/dense_5/kernel/m
?
FAdam/credict_fraud_detect/decoder/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_5/kernel/m*
_output_shapes

:*
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/credict_fraud_detect/decoder/dense_5/bias/m
?
DAdam/credict_fraud_detect/decoder/dense_5/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_5/bias/m*
_output_shapes
:*
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42Adam/credict_fraud_detect/decoder/dense_6/kernel/m
?
FAdam/credict_fraud_detect/decoder/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_6/kernel/m*
_output_shapes

: *
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/credict_fraud_detect/decoder/dense_6/bias/m
?
DAdam/credict_fraud_detect/decoder/dense_6/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_6/bias/m*
_output_shapes
: *
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*C
shared_name42Adam/credict_fraud_detect/decoder/dense_7/kernel/m
?
FAdam/credict_fraud_detect/decoder/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_7/kernel/m*
_output_shapes

: @*
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/credict_fraud_detect/decoder/dense_7/bias/m
?
DAdam/credict_fraud_detect/decoder/dense_7/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_7/bias/m*
_output_shapes
:@*
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*C
shared_name42Adam/credict_fraud_detect/decoder/dense_8/kernel/m
?
FAdam/credict_fraud_detect/decoder/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_8/kernel/m*
_output_shapes
:	@?*
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/credict_fraud_detect/decoder/dense_8/bias/m
?
DAdam/credict_fraud_detect/decoder/dense_8/bias/m/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_8/bias/m*
_output_shapes	
:?*
dtype0
?
*Adam/credict_fraud_detect/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/credict_fraud_detect/dense_9/kernel/m
?
>Adam/credict_fraud_detect/dense_9/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/credict_fraud_detect/dense_9/kernel/m*
_output_shapes
:	?*
dtype0
?
(Adam/credict_fraud_detect/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/credict_fraud_detect/dense_9/bias/m
?
<Adam/credict_fraud_detect/dense_9/bias/m/Read/ReadVariableOpReadVariableOp(Adam/credict_fraud_detect/dense_9/bias/m*
_output_shapes
:*
dtype0
?
0Adam/credict_fraud_detect/encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*A
shared_name20Adam/credict_fraud_detect/encoder/dense/kernel/v
?
DAdam/credict_fraud_detect/encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
.Adam/credict_fraud_detect/encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/credict_fraud_detect/encoder/dense/bias/v
?
BAdam/credict_fraud_detect/encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp.Adam/credict_fraud_detect/encoder/dense/bias/v*
_output_shapes	
:?*
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*C
shared_name42Adam/credict_fraud_detect/encoder/dense_1/kernel/v
?
FAdam/credict_fraud_detect/encoder/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_1/kernel/v*
_output_shapes
:	?@*
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/credict_fraud_detect/encoder/dense_1/bias/v
?
DAdam/credict_fraud_detect/encoder/dense_1/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_1/bias/v*
_output_shapes
:@*
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *C
shared_name42Adam/credict_fraud_detect/encoder/dense_2/kernel/v
?
FAdam/credict_fraud_detect/encoder/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_2/kernel/v*
_output_shapes

:@ *
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/credict_fraud_detect/encoder/dense_2/bias/v
?
DAdam/credict_fraud_detect/encoder/dense_2/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_2/bias/v*
_output_shapes
: *
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42Adam/credict_fraud_detect/encoder/dense_3/kernel/v
?
FAdam/credict_fraud_detect/encoder/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_3/kernel/v*
_output_shapes

: *
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/credict_fraud_detect/encoder/dense_3/bias/v
?
DAdam/credict_fraud_detect/encoder/dense_3/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_3/bias/v*
_output_shapes
:*
dtype0
?
2Adam/credict_fraud_detect/encoder/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/credict_fraud_detect/encoder/dense_4/kernel/v
?
FAdam/credict_fraud_detect/encoder/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/encoder/dense_4/kernel/v*
_output_shapes

:*
dtype0
?
0Adam/credict_fraud_detect/encoder/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/credict_fraud_detect/encoder/dense_4/bias/v
?
DAdam/credict_fraud_detect/encoder/dense_4/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/encoder/dense_4/bias/v*
_output_shapes
:*
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adam/credict_fraud_detect/decoder/dense_5/kernel/v
?
FAdam/credict_fraud_detect/decoder/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_5/kernel/v*
_output_shapes

:*
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/credict_fraud_detect/decoder/dense_5/bias/v
?
DAdam/credict_fraud_detect/decoder/dense_5/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_5/bias/v*
_output_shapes
:*
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42Adam/credict_fraud_detect/decoder/dense_6/kernel/v
?
FAdam/credict_fraud_detect/decoder/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_6/kernel/v*
_output_shapes

: *
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/credict_fraud_detect/decoder/dense_6/bias/v
?
DAdam/credict_fraud_detect/decoder/dense_6/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_6/bias/v*
_output_shapes
: *
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*C
shared_name42Adam/credict_fraud_detect/decoder/dense_7/kernel/v
?
FAdam/credict_fraud_detect/decoder/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_7/kernel/v*
_output_shapes

: @*
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adam/credict_fraud_detect/decoder/dense_7/bias/v
?
DAdam/credict_fraud_detect/decoder/dense_7/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_7/bias/v*
_output_shapes
:@*
dtype0
?
2Adam/credict_fraud_detect/decoder/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*C
shared_name42Adam/credict_fraud_detect/decoder/dense_8/kernel/v
?
FAdam/credict_fraud_detect/decoder/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/credict_fraud_detect/decoder/dense_8/kernel/v*
_output_shapes
:	@?*
dtype0
?
0Adam/credict_fraud_detect/decoder/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/credict_fraud_detect/decoder/dense_8/bias/v
?
DAdam/credict_fraud_detect/decoder/dense_8/bias/v/Read/ReadVariableOpReadVariableOp0Adam/credict_fraud_detect/decoder/dense_8/bias/v*
_output_shapes	
:?*
dtype0
?
*Adam/credict_fraud_detect/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/credict_fraud_detect/dense_9/kernel/v
?
>Adam/credict_fraud_detect/dense_9/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/credict_fraud_detect/dense_9/kernel/v*
_output_shapes
:	?*
dtype0
?
(Adam/credict_fraud_detect/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/credict_fraud_detect/dense_9/bias/v
?
<Adam/credict_fraud_detect/dense_9/bias/v/Read/ReadVariableOpReadVariableOp(Adam/credict_fraud_detect/dense_9/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?z
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?z
value?zB?z B?z
?
layer-0
layer_with_weights-0
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
w
	encoder

decoder
	dense
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratem?m?m?m?m?m?m?m?m?m?m? m?!m?"m?#m?$m?%m?&m?'m?(m?v?v?v?v?v?v?v?v?v?v?v? v?!v?"v?#v?$v?%v?&v?'v?(v?
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
 
?
)metrics

*layers
+layer_regularization_losses
,layer_metrics
	variables
-non_trainable_variables
trainable_variables
regularization_losses
 
?
.	dense_128
/dense_64
0dense_32
1dense_8
2dense_4
3regularization_losses
4	variables
5trainable_variables
6	keras_api
?
7dense_8
8dense_32
9dense_64
:	dense_128
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

'kernel
(bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19
 
?
Cmetrics

Dlayers
Elayer_regularization_losses
Flayer_metrics
	variables
Gnon_trainable_variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)credict_fraud_detect/encoder/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'credict_fraud_detect/encoder/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+credict_fraud_detect/encoder/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)credict_fraud_detect/encoder/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+credict_fraud_detect/encoder/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)credict_fraud_detect/encoder/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+credict_fraud_detect/encoder/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)credict_fraud_detect/encoder/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+credict_fraud_detect/encoder/dense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)credict_fraud_detect/encoder/dense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+credict_fraud_detect/decoder/dense_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)credict_fraud_detect/decoder/dense_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+credict_fraud_detect/decoder/dense_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)credict_fraud_detect/decoder/dense_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+credict_fraud_detect/decoder/dense_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)credict_fraud_detect/decoder/dense_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+credict_fraud_detect/decoder/dense_8/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)credict_fraud_detect/decoder/dense_8/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#credict_fraud_detect/dense_9/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!credict_fraud_detect/dense_9/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
J2
K3

0
1
 
 
 
h

kernel
bias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

kernel
bias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
h

kernel
bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
h

kernel
bias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
h

kernel
bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
 
F
0
1
2
3
4
5
6
7
8
9
F
0
1
2
3
4
5
6
7
8
9
?
3regularization_losses

`layers
alayer_regularization_losses
blayer_metrics
4	variables
cnon_trainable_variables
5trainable_variables
dmetrics
h

kernel
 bias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
h

!kernel
"bias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
h

#kernel
$bias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
h

%kernel
&bias
qregularization_losses
r	variables
strainable_variables
t	keras_api
 
8
0
 1
!2
"3
#4
$5
%6
&7
8
0
 1
!2
"3
#4
$5
%6
&7
?
;regularization_losses

ulayers
vlayer_regularization_losses
wlayer_metrics
<	variables
xnon_trainable_variables
=trainable_variables
ymetrics
 

'0
(1

'0
(1
?
?regularization_losses

zlayers
{layer_regularization_losses
|layer_metrics
@	variables
}non_trainable_variables
Atrainable_variables
~metrics
 

	0

1
2
 
 
 
7
	total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
v
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api
 

0
1

0
1
?
Lregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
M	variables
?non_trainable_variables
Ntrainable_variables
?metrics
 

0
1

0
1
?
Pregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Q	variables
?non_trainable_variables
Rtrainable_variables
?metrics
 

0
1

0
1
?
Tregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
U	variables
?non_trainable_variables
Vtrainable_variables
?metrics
 

0
1

0
1
?
Xregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Y	variables
?non_trainable_variables
Ztrainable_variables
?metrics
 

0
1

0
1
?
\regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
]	variables
?non_trainable_variables
^trainable_variables
?metrics
#
.0
/1
02
13
24
 
 
 
 
 

0
 1

0
 1
?
eregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
f	variables
?non_trainable_variables
gtrainable_variables
?metrics
 

!0
"1

!0
"1
?
iregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
j	variables
?non_trainable_variables
ktrainable_variables
?metrics
 

#0
$1

#0
$1
?
mregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
n	variables
?non_trainable_variables
otrainable_variables
?metrics
 

%0
&1

%0
&1
?
qregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
r	variables
?non_trainable_variables
strainable_variables
?metrics

70
81
92
:3
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/credict_fraud_detect/encoder/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_7/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_7/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_8/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_8/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/credict_fraud_detect/dense_9/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/credict_fraud_detect/dense_9/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/credict_fraud_detect/encoder/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/encoder/dense_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/encoder/dense_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_7/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_7/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/credict_fraud_detect/decoder/dense_8/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/credict_fraud_detect/decoder/dense_8/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/credict_fraud_detect/dense_9/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/credict_fraud_detect/dense_9/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)credict_fraud_detect/encoder/dense/kernel'credict_fraud_detect/encoder/dense/bias+credict_fraud_detect/encoder/dense_1/kernel)credict_fraud_detect/encoder/dense_1/bias+credict_fraud_detect/encoder/dense_2/kernel)credict_fraud_detect/encoder/dense_2/bias+credict_fraud_detect/encoder/dense_3/kernel)credict_fraud_detect/encoder/dense_3/bias+credict_fraud_detect/encoder/dense_4/kernel)credict_fraud_detect/encoder/dense_4/bias+credict_fraud_detect/decoder/dense_5/kernel)credict_fraud_detect/decoder/dense_5/bias+credict_fraud_detect/decoder/dense_6/kernel)credict_fraud_detect/decoder/dense_6/bias+credict_fraud_detect/decoder/dense_7/kernel)credict_fraud_detect/decoder/dense_7/bias+credict_fraud_detect/decoder/dense_8/kernel)credict_fraud_detect/decoder/dense_8/bias#credict_fraud_detect/dense_9/kernel!credict_fraud_detect/dense_9/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_3842412
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp=credict_fraud_detect/encoder/dense/kernel/Read/ReadVariableOp;credict_fraud_detect/encoder/dense/bias/Read/ReadVariableOp?credict_fraud_detect/encoder/dense_1/kernel/Read/ReadVariableOp=credict_fraud_detect/encoder/dense_1/bias/Read/ReadVariableOp?credict_fraud_detect/encoder/dense_2/kernel/Read/ReadVariableOp=credict_fraud_detect/encoder/dense_2/bias/Read/ReadVariableOp?credict_fraud_detect/encoder/dense_3/kernel/Read/ReadVariableOp=credict_fraud_detect/encoder/dense_3/bias/Read/ReadVariableOp?credict_fraud_detect/encoder/dense_4/kernel/Read/ReadVariableOp=credict_fraud_detect/encoder/dense_4/bias/Read/ReadVariableOp?credict_fraud_detect/decoder/dense_5/kernel/Read/ReadVariableOp=credict_fraud_detect/decoder/dense_5/bias/Read/ReadVariableOp?credict_fraud_detect/decoder/dense_6/kernel/Read/ReadVariableOp=credict_fraud_detect/decoder/dense_6/bias/Read/ReadVariableOp?credict_fraud_detect/decoder/dense_7/kernel/Read/ReadVariableOp=credict_fraud_detect/decoder/dense_7/bias/Read/ReadVariableOp?credict_fraud_detect/decoder/dense_8/kernel/Read/ReadVariableOp=credict_fraud_detect/decoder/dense_8/bias/Read/ReadVariableOp7credict_fraud_detect/dense_9/kernel/Read/ReadVariableOp5credict_fraud_detect/dense_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense/kernel/m/Read/ReadVariableOpBAdam/credict_fraud_detect/encoder/dense/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_1/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_1/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_2/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_2/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_3/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_3/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_4/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_4/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_5/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_5/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_6/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_6/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_7/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_7/bias/m/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_8/kernel/m/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_8/bias/m/Read/ReadVariableOp>Adam/credict_fraud_detect/dense_9/kernel/m/Read/ReadVariableOp<Adam/credict_fraud_detect/dense_9/bias/m/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense/kernel/v/Read/ReadVariableOpBAdam/credict_fraud_detect/encoder/dense/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_1/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_1/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_2/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_2/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_3/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_3/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/encoder/dense_4/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/encoder/dense_4/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_5/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_5/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_6/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_6/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_7/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_7/bias/v/Read/ReadVariableOpFAdam/credict_fraud_detect/decoder/dense_8/kernel/v/Read/ReadVariableOpDAdam/credict_fraud_detect/decoder/dense_8/bias/v/Read/ReadVariableOp>Adam/credict_fraud_detect/dense_9/kernel/v/Read/ReadVariableOp<Adam/credict_fraud_detect/dense_9/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_3843008
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate)credict_fraud_detect/encoder/dense/kernel'credict_fraud_detect/encoder/dense/bias+credict_fraud_detect/encoder/dense_1/kernel)credict_fraud_detect/encoder/dense_1/bias+credict_fraud_detect/encoder/dense_2/kernel)credict_fraud_detect/encoder/dense_2/bias+credict_fraud_detect/encoder/dense_3/kernel)credict_fraud_detect/encoder/dense_3/bias+credict_fraud_detect/encoder/dense_4/kernel)credict_fraud_detect/encoder/dense_4/bias+credict_fraud_detect/decoder/dense_5/kernel)credict_fraud_detect/decoder/dense_5/bias+credict_fraud_detect/decoder/dense_6/kernel)credict_fraud_detect/decoder/dense_6/bias+credict_fraud_detect/decoder/dense_7/kernel)credict_fraud_detect/decoder/dense_7/bias+credict_fraud_detect/decoder/dense_8/kernel)credict_fraud_detect/decoder/dense_8/bias#credict_fraud_detect/dense_9/kernel!credict_fraud_detect/dense_9/biastotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_negatives_10Adam/credict_fraud_detect/encoder/dense/kernel/m.Adam/credict_fraud_detect/encoder/dense/bias/m2Adam/credict_fraud_detect/encoder/dense_1/kernel/m0Adam/credict_fraud_detect/encoder/dense_1/bias/m2Adam/credict_fraud_detect/encoder/dense_2/kernel/m0Adam/credict_fraud_detect/encoder/dense_2/bias/m2Adam/credict_fraud_detect/encoder/dense_3/kernel/m0Adam/credict_fraud_detect/encoder/dense_3/bias/m2Adam/credict_fraud_detect/encoder/dense_4/kernel/m0Adam/credict_fraud_detect/encoder/dense_4/bias/m2Adam/credict_fraud_detect/decoder/dense_5/kernel/m0Adam/credict_fraud_detect/decoder/dense_5/bias/m2Adam/credict_fraud_detect/decoder/dense_6/kernel/m0Adam/credict_fraud_detect/decoder/dense_6/bias/m2Adam/credict_fraud_detect/decoder/dense_7/kernel/m0Adam/credict_fraud_detect/decoder/dense_7/bias/m2Adam/credict_fraud_detect/decoder/dense_8/kernel/m0Adam/credict_fraud_detect/decoder/dense_8/bias/m*Adam/credict_fraud_detect/dense_9/kernel/m(Adam/credict_fraud_detect/dense_9/bias/m0Adam/credict_fraud_detect/encoder/dense/kernel/v.Adam/credict_fraud_detect/encoder/dense/bias/v2Adam/credict_fraud_detect/encoder/dense_1/kernel/v0Adam/credict_fraud_detect/encoder/dense_1/bias/v2Adam/credict_fraud_detect/encoder/dense_2/kernel/v0Adam/credict_fraud_detect/encoder/dense_2/bias/v2Adam/credict_fraud_detect/encoder/dense_3/kernel/v0Adam/credict_fraud_detect/encoder/dense_3/bias/v2Adam/credict_fraud_detect/encoder/dense_4/kernel/v0Adam/credict_fraud_detect/encoder/dense_4/bias/v2Adam/credict_fraud_detect/decoder/dense_5/kernel/v0Adam/credict_fraud_detect/decoder/dense_5/bias/v2Adam/credict_fraud_detect/decoder/dense_6/kernel/v0Adam/credict_fraud_detect/decoder/dense_6/bias/v2Adam/credict_fraud_detect/decoder/dense_7/kernel/v0Adam/credict_fraud_detect/decoder/dense_7/bias/v2Adam/credict_fraud_detect/decoder/dense_8/kernel/v0Adam/credict_fraud_detect/decoder/dense_8/bias/v*Adam/credict_fraud_detect/dense_9/kernel/v(Adam/credict_fraud_detect/dense_9/bias/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_3843243??

??
?
B__inference_model_layer_call_and_return_conditional_losses_3842632

inputsU
Acredict_fraud_detect_encoder_dense_matmul_readvariableop_resource:
??Q
Bcredict_fraud_detect_encoder_dense_biasadd_readvariableop_resource:	?V
Ccredict_fraud_detect_encoder_dense_1_matmul_readvariableop_resource:	?@R
Dcredict_fraud_detect_encoder_dense_1_biasadd_readvariableop_resource:@U
Ccredict_fraud_detect_encoder_dense_2_matmul_readvariableop_resource:@ R
Dcredict_fraud_detect_encoder_dense_2_biasadd_readvariableop_resource: U
Ccredict_fraud_detect_encoder_dense_3_matmul_readvariableop_resource: R
Dcredict_fraud_detect_encoder_dense_3_biasadd_readvariableop_resource:U
Ccredict_fraud_detect_encoder_dense_4_matmul_readvariableop_resource:R
Dcredict_fraud_detect_encoder_dense_4_biasadd_readvariableop_resource:U
Ccredict_fraud_detect_decoder_dense_5_matmul_readvariableop_resource:R
Dcredict_fraud_detect_decoder_dense_5_biasadd_readvariableop_resource:U
Ccredict_fraud_detect_decoder_dense_6_matmul_readvariableop_resource: R
Dcredict_fraud_detect_decoder_dense_6_biasadd_readvariableop_resource: U
Ccredict_fraud_detect_decoder_dense_7_matmul_readvariableop_resource: @R
Dcredict_fraud_detect_decoder_dense_7_biasadd_readvariableop_resource:@V
Ccredict_fraud_detect_decoder_dense_8_matmul_readvariableop_resource:	@?S
Dcredict_fraud_detect_decoder_dense_8_biasadd_readvariableop_resource:	?N
;credict_fraud_detect_dense_9_matmul_readvariableop_resource:	?J
<credict_fraud_detect_dense_9_biasadd_readvariableop_resource:
identity??;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp?;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp?;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp?;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp?3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp?2credict_fraud_detect/dense_9/MatMul/ReadVariableOp?9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp?
8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOpReadVariableOpAcredict_fraud_detect_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp?
)credict_fraud_detect/encoder/dense/MatMulMatMulinputs@credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)credict_fraud_detect/encoder/dense/MatMul?
9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOpReadVariableOpBcredict_fraud_detect_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?
*credict_fraud_detect/encoder/dense/BiasAddBiasAdd3credict_fraud_detect/encoder/dense/MatMul:product:0Acredict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*credict_fraud_detect/encoder/dense/BiasAdd?
:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02<
:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_1/MatMulMatMul3credict_fraud_detect/encoder/dense/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2-
+credict_fraud_detect/encoder/dense_1/MatMul?
;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_1/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_1/MatMul:product:0Ccredict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2.
,credict_fraud_detect/encoder/dense_1/BiasAdd?
:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02<
:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_2/MatMulMatMul5credict_fraud_detect/encoder/dense_1/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2-
+credict_fraud_detect/encoder/dense_2/MatMul?
;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_2/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_2/MatMul:product:0Ccredict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2.
,credict_fraud_detect/encoder/dense_2/BiasAdd?
:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02<
:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_3/MatMulMatMul5credict_fraud_detect/encoder/dense_2/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+credict_fraud_detect/encoder/dense_3/MatMul?
;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_3/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_3/MatMul:product:0Ccredict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,credict_fraud_detect/encoder/dense_3/BiasAdd?
:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_4/MatMulMatMul5credict_fraud_detect/encoder/dense_3/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+credict_fraud_detect/encoder/dense_4/MatMul?
;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_4/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_4/MatMul:product:0Ccredict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,credict_fraud_detect/encoder/dense_4/BiasAdd?
:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_5/MatMulMatMul5credict_fraud_detect/encoder/dense_4/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+credict_fraud_detect/decoder/dense_5/MatMul?
;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_5/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_5/MatMul:product:0Ccredict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,credict_fraud_detect/decoder/dense_5/BiasAdd?
:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02<
:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_6/MatMulMatMul5credict_fraud_detect/decoder/dense_5/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2-
+credict_fraud_detect/decoder/dense_6/MatMul?
;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_6/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_6/MatMul:product:0Ccredict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2.
,credict_fraud_detect/decoder/dense_6/BiasAdd?
:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_7_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02<
:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_7/MatMulMatMul5credict_fraud_detect/decoder/dense_6/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2-
+credict_fraud_detect/decoder/dense_7/MatMul?
;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_7/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_7/MatMul:product:0Ccredict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2.
,credict_fraud_detect/decoder/dense_7/BiasAdd?
:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_8_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02<
:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_8/MatMulMatMul5credict_fraud_detect/decoder/dense_7/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+credict_fraud_detect/decoder/dense_8/MatMul?
;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_8/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_8/MatMul:product:0Ccredict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,credict_fraud_detect/decoder/dense_8/BiasAdd?
2credict_fraud_detect/dense_9/MatMul/ReadVariableOpReadVariableOp;credict_fraud_detect_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype024
2credict_fraud_detect/dense_9/MatMul/ReadVariableOp?
#credict_fraud_detect/dense_9/MatMulMatMul5credict_fraud_detect/decoder/dense_8/BiasAdd:output:0:credict_fraud_detect/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#credict_fraud_detect/dense_9/MatMul?
3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOpReadVariableOp<credict_fraud_detect_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp?
$credict_fraud_detect/dense_9/BiasAddBiasAdd-credict_fraud_detect/dense_9/MatMul:product:0;credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$credict_fraud_detect/dense_9/BiasAdd?
$credict_fraud_detect/dense_9/SoftmaxSoftmax-credict_fraud_detect/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2&
$credict_fraud_detect/dense_9/Softmax?

IdentityIdentity.credict_fraud_detect/dense_9/Softmax:softmax:0<^credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp<^credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp<^credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp<^credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp4^credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp3^credict_fraud_detect/dense_9/MatMul/ReadVariableOp:^credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp9^credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2z
;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp2z
;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp2z
;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp2z
;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp2j
3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp2h
2credict_fraud_detect/dense_9/MatMul/ReadVariableOp2credict_fraud_detect/dense_9/MatMul/ReadVariableOp2v
9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp2t
8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?#
?
D__inference_decoder_layer_call_and_return_conditional_losses_3841857

inputs8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: @5
'dense_7_biasadd_readvariableop_resource:@9
&dense_8_matmul_readvariableop_resource:	@?6
'dense_8_biasadd_readvariableop_resource:	?
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/BiasAdd:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_6/BiasAdd?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/BiasAdd:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAdd?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldense_7/BiasAdd:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAdd?
IdentityIdentitydense_8/BiasAdd:output:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_model_layer_call_and_return_conditional_losses_3842567

inputsU
Acredict_fraud_detect_encoder_dense_matmul_readvariableop_resource:
??Q
Bcredict_fraud_detect_encoder_dense_biasadd_readvariableop_resource:	?V
Ccredict_fraud_detect_encoder_dense_1_matmul_readvariableop_resource:	?@R
Dcredict_fraud_detect_encoder_dense_1_biasadd_readvariableop_resource:@U
Ccredict_fraud_detect_encoder_dense_2_matmul_readvariableop_resource:@ R
Dcredict_fraud_detect_encoder_dense_2_biasadd_readvariableop_resource: U
Ccredict_fraud_detect_encoder_dense_3_matmul_readvariableop_resource: R
Dcredict_fraud_detect_encoder_dense_3_biasadd_readvariableop_resource:U
Ccredict_fraud_detect_encoder_dense_4_matmul_readvariableop_resource:R
Dcredict_fraud_detect_encoder_dense_4_biasadd_readvariableop_resource:U
Ccredict_fraud_detect_decoder_dense_5_matmul_readvariableop_resource:R
Dcredict_fraud_detect_decoder_dense_5_biasadd_readvariableop_resource:U
Ccredict_fraud_detect_decoder_dense_6_matmul_readvariableop_resource: R
Dcredict_fraud_detect_decoder_dense_6_biasadd_readvariableop_resource: U
Ccredict_fraud_detect_decoder_dense_7_matmul_readvariableop_resource: @R
Dcredict_fraud_detect_decoder_dense_7_biasadd_readvariableop_resource:@V
Ccredict_fraud_detect_decoder_dense_8_matmul_readvariableop_resource:	@?S
Dcredict_fraud_detect_decoder_dense_8_biasadd_readvariableop_resource:	?N
;credict_fraud_detect_dense_9_matmul_readvariableop_resource:	?J
<credict_fraud_detect_dense_9_biasadd_readvariableop_resource:
identity??;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp?;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp?;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp?;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp?:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp?3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp?2credict_fraud_detect/dense_9/MatMul/ReadVariableOp?9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp?;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp?:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp?
8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOpReadVariableOpAcredict_fraud_detect_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp?
)credict_fraud_detect/encoder/dense/MatMulMatMulinputs@credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)credict_fraud_detect/encoder/dense/MatMul?
9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOpReadVariableOpBcredict_fraud_detect_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?
*credict_fraud_detect/encoder/dense/BiasAddBiasAdd3credict_fraud_detect/encoder/dense/MatMul:product:0Acredict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*credict_fraud_detect/encoder/dense/BiasAdd?
:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02<
:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_1/MatMulMatMul3credict_fraud_detect/encoder/dense/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2-
+credict_fraud_detect/encoder/dense_1/MatMul?
;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_1/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_1/MatMul:product:0Ccredict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2.
,credict_fraud_detect/encoder/dense_1/BiasAdd?
:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02<
:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_2/MatMulMatMul5credict_fraud_detect/encoder/dense_1/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2-
+credict_fraud_detect/encoder/dense_2/MatMul?
;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_2/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_2/MatMul:product:0Ccredict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2.
,credict_fraud_detect/encoder/dense_2/BiasAdd?
:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02<
:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_3/MatMulMatMul5credict_fraud_detect/encoder/dense_2/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+credict_fraud_detect/encoder/dense_3/MatMul?
;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_3/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_3/MatMul:product:0Ccredict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,credict_fraud_detect/encoder/dense_3/BiasAdd?
:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp?
+credict_fraud_detect/encoder/dense_4/MatMulMatMul5credict_fraud_detect/encoder/dense_3/BiasAdd:output:0Bcredict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+credict_fraud_detect/encoder/dense_4/MatMul?
;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp?
,credict_fraud_detect/encoder/dense_4/BiasAddBiasAdd5credict_fraud_detect/encoder/dense_4/MatMul:product:0Ccredict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,credict_fraud_detect/encoder/dense_4/BiasAdd?
:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_5/MatMulMatMul5credict_fraud_detect/encoder/dense_4/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+credict_fraud_detect/decoder/dense_5/MatMul?
;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_5/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_5/MatMul:product:0Ccredict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,credict_fraud_detect/decoder/dense_5/BiasAdd?
:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02<
:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_6/MatMulMatMul5credict_fraud_detect/decoder/dense_5/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2-
+credict_fraud_detect/decoder/dense_6/MatMul?
;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_6/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_6/MatMul:product:0Ccredict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2.
,credict_fraud_detect/decoder/dense_6/BiasAdd?
:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_7_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02<
:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_7/MatMulMatMul5credict_fraud_detect/decoder/dense_6/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2-
+credict_fraud_detect/decoder/dense_7/MatMul?
;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_7/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_7/MatMul:product:0Ccredict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2.
,credict_fraud_detect/decoder/dense_7/BiasAdd?
:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOpReadVariableOpCcredict_fraud_detect_decoder_dense_8_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02<
:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp?
+credict_fraud_detect/decoder/dense_8/MatMulMatMul5credict_fraud_detect/decoder/dense_7/BiasAdd:output:0Bcredict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+credict_fraud_detect/decoder/dense_8/MatMul?
;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOpReadVariableOpDcredict_fraud_detect_decoder_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp?
,credict_fraud_detect/decoder/dense_8/BiasAddBiasAdd5credict_fraud_detect/decoder/dense_8/MatMul:product:0Ccredict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,credict_fraud_detect/decoder/dense_8/BiasAdd?
2credict_fraud_detect/dense_9/MatMul/ReadVariableOpReadVariableOp;credict_fraud_detect_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype024
2credict_fraud_detect/dense_9/MatMul/ReadVariableOp?
#credict_fraud_detect/dense_9/MatMulMatMul5credict_fraud_detect/decoder/dense_8/BiasAdd:output:0:credict_fraud_detect/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#credict_fraud_detect/dense_9/MatMul?
3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOpReadVariableOp<credict_fraud_detect_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp?
$credict_fraud_detect/dense_9/BiasAddBiasAdd-credict_fraud_detect/dense_9/MatMul:product:0;credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$credict_fraud_detect/dense_9/BiasAdd?
$credict_fraud_detect/dense_9/SoftmaxSoftmax-credict_fraud_detect/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2&
$credict_fraud_detect/dense_9/Softmax?

IdentityIdentity.credict_fraud_detect/dense_9/Softmax:softmax:0<^credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp<^credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp<^credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp<^credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp;^credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp4^credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp3^credict_fraud_detect/dense_9/MatMul/ReadVariableOp:^credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp9^credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp<^credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp;^credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2z
;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp2z
;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp2z
;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp2z
;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp;credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp:credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp2j
3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp3credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp2h
2credict_fraud_detect/dense_9/MatMul/ReadVariableOp2credict_fraud_detect/dense_9/MatMul/ReadVariableOp2v
9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp9credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp2t
8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp8credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp2z
;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp;credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp2x
:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_credict_fraud_detect_layer_call_fn_3841939
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:	@?

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_38418932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
B__inference_model_layer_call_and_return_conditional_losses_3842359
input_10
credict_fraud_detect_3842317:
??+
credict_fraud_detect_3842319:	?/
credict_fraud_detect_3842321:	?@*
credict_fraud_detect_3842323:@.
credict_fraud_detect_3842325:@ *
credict_fraud_detect_3842327: .
credict_fraud_detect_3842329: *
credict_fraud_detect_3842331:.
credict_fraud_detect_3842333:*
credict_fraud_detect_3842335:.
credict_fraud_detect_3842337:*
credict_fraud_detect_3842339:.
credict_fraud_detect_3842341: *
credict_fraud_detect_3842343: .
credict_fraud_detect_3842345: @*
credict_fraud_detect_3842347:@/
credict_fraud_detect_3842349:	@?+
credict_fraud_detect_3842351:	?/
credict_fraud_detect_3842353:	?*
credict_fraud_detect_3842355:
identity??,credict_fraud_detect/StatefulPartitionedCall?
,credict_fraud_detect/StatefulPartitionedCallStatefulPartitionedCallinput_1credict_fraud_detect_3842317credict_fraud_detect_3842319credict_fraud_detect_3842321credict_fraud_detect_3842323credict_fraud_detect_3842325credict_fraud_detect_3842327credict_fraud_detect_3842329credict_fraud_detect_3842331credict_fraud_detect_3842333credict_fraud_detect_3842335credict_fraud_detect_3842337credict_fraud_detect_3842339credict_fraud_detect_3842341credict_fraud_detect_3842343credict_fraud_detect_3842345credict_fraud_detect_3842347credict_fraud_detect_3842349credict_fraud_detect_3842351credict_fraud_detect_3842353credict_fraud_detect_3842355* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_38418932.
,credict_fraud_detect/StatefulPartitionedCall?
IdentityIdentity5credict_fraud_detect/StatefulPartitionedCall:output:0-^credict_fraud_detect/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2\
,credict_fraud_detect/StatefulPartitionedCall,credict_fraud_detect/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
??
?9
#__inference__traced_restore_3843243
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: P
<assignvariableop_5_credict_fraud_detect_encoder_dense_kernel:
??I
:assignvariableop_6_credict_fraud_detect_encoder_dense_bias:	?Q
>assignvariableop_7_credict_fraud_detect_encoder_dense_1_kernel:	?@J
<assignvariableop_8_credict_fraud_detect_encoder_dense_1_bias:@P
>assignvariableop_9_credict_fraud_detect_encoder_dense_2_kernel:@ K
=assignvariableop_10_credict_fraud_detect_encoder_dense_2_bias: Q
?assignvariableop_11_credict_fraud_detect_encoder_dense_3_kernel: K
=assignvariableop_12_credict_fraud_detect_encoder_dense_3_bias:Q
?assignvariableop_13_credict_fraud_detect_encoder_dense_4_kernel:K
=assignvariableop_14_credict_fraud_detect_encoder_dense_4_bias:Q
?assignvariableop_15_credict_fraud_detect_decoder_dense_5_kernel:K
=assignvariableop_16_credict_fraud_detect_decoder_dense_5_bias:Q
?assignvariableop_17_credict_fraud_detect_decoder_dense_6_kernel: K
=assignvariableop_18_credict_fraud_detect_decoder_dense_6_bias: Q
?assignvariableop_19_credict_fraud_detect_decoder_dense_7_kernel: @K
=assignvariableop_20_credict_fraud_detect_decoder_dense_7_bias:@R
?assignvariableop_21_credict_fraud_detect_decoder_dense_8_kernel:	@?L
=assignvariableop_22_credict_fraud_detect_decoder_dense_8_bias:	?J
7assignvariableop_23_credict_fraud_detect_dense_9_kernel:	?C
5assignvariableop_24_credict_fraud_detect_dense_9_bias:#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: 1
"assignvariableop_29_true_positives:	?1
"assignvariableop_30_true_negatives:	?2
#assignvariableop_31_false_positives:	?2
#assignvariableop_32_false_negatives:	?2
$assignvariableop_33_true_positives_1:3
%assignvariableop_34_false_negatives_1:X
Dassignvariableop_35_adam_credict_fraud_detect_encoder_dense_kernel_m:
??Q
Bassignvariableop_36_adam_credict_fraud_detect_encoder_dense_bias_m:	?Y
Fassignvariableop_37_adam_credict_fraud_detect_encoder_dense_1_kernel_m:	?@R
Dassignvariableop_38_adam_credict_fraud_detect_encoder_dense_1_bias_m:@X
Fassignvariableop_39_adam_credict_fraud_detect_encoder_dense_2_kernel_m:@ R
Dassignvariableop_40_adam_credict_fraud_detect_encoder_dense_2_bias_m: X
Fassignvariableop_41_adam_credict_fraud_detect_encoder_dense_3_kernel_m: R
Dassignvariableop_42_adam_credict_fraud_detect_encoder_dense_3_bias_m:X
Fassignvariableop_43_adam_credict_fraud_detect_encoder_dense_4_kernel_m:R
Dassignvariableop_44_adam_credict_fraud_detect_encoder_dense_4_bias_m:X
Fassignvariableop_45_adam_credict_fraud_detect_decoder_dense_5_kernel_m:R
Dassignvariableop_46_adam_credict_fraud_detect_decoder_dense_5_bias_m:X
Fassignvariableop_47_adam_credict_fraud_detect_decoder_dense_6_kernel_m: R
Dassignvariableop_48_adam_credict_fraud_detect_decoder_dense_6_bias_m: X
Fassignvariableop_49_adam_credict_fraud_detect_decoder_dense_7_kernel_m: @R
Dassignvariableop_50_adam_credict_fraud_detect_decoder_dense_7_bias_m:@Y
Fassignvariableop_51_adam_credict_fraud_detect_decoder_dense_8_kernel_m:	@?S
Dassignvariableop_52_adam_credict_fraud_detect_decoder_dense_8_bias_m:	?Q
>assignvariableop_53_adam_credict_fraud_detect_dense_9_kernel_m:	?J
<assignvariableop_54_adam_credict_fraud_detect_dense_9_bias_m:X
Dassignvariableop_55_adam_credict_fraud_detect_encoder_dense_kernel_v:
??Q
Bassignvariableop_56_adam_credict_fraud_detect_encoder_dense_bias_v:	?Y
Fassignvariableop_57_adam_credict_fraud_detect_encoder_dense_1_kernel_v:	?@R
Dassignvariableop_58_adam_credict_fraud_detect_encoder_dense_1_bias_v:@X
Fassignvariableop_59_adam_credict_fraud_detect_encoder_dense_2_kernel_v:@ R
Dassignvariableop_60_adam_credict_fraud_detect_encoder_dense_2_bias_v: X
Fassignvariableop_61_adam_credict_fraud_detect_encoder_dense_3_kernel_v: R
Dassignvariableop_62_adam_credict_fraud_detect_encoder_dense_3_bias_v:X
Fassignvariableop_63_adam_credict_fraud_detect_encoder_dense_4_kernel_v:R
Dassignvariableop_64_adam_credict_fraud_detect_encoder_dense_4_bias_v:X
Fassignvariableop_65_adam_credict_fraud_detect_decoder_dense_5_kernel_v:R
Dassignvariableop_66_adam_credict_fraud_detect_decoder_dense_5_bias_v:X
Fassignvariableop_67_adam_credict_fraud_detect_decoder_dense_6_kernel_v: R
Dassignvariableop_68_adam_credict_fraud_detect_decoder_dense_6_bias_v: X
Fassignvariableop_69_adam_credict_fraud_detect_decoder_dense_7_kernel_v: @R
Dassignvariableop_70_adam_credict_fraud_detect_decoder_dense_7_bias_v:@Y
Fassignvariableop_71_adam_credict_fraud_detect_decoder_dense_8_kernel_v:	@?S
Dassignvariableop_72_adam_credict_fraud_detect_decoder_dense_8_bias_v:	?Q
>assignvariableop_73_adam_credict_fraud_detect_dense_9_kernel_v:	?J
<assignvariableop_74_adam_credict_fraud_detect_dense_9_bias_v:
identity_76??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_8?AssignVariableOp_9?#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?"
value?"B?"LB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp<assignvariableop_5_credict_fraud_detect_encoder_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp:assignvariableop_6_credict_fraud_detect_encoder_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp>assignvariableop_7_credict_fraud_detect_encoder_dense_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp<assignvariableop_8_credict_fraud_detect_encoder_dense_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp>assignvariableop_9_credict_fraud_detect_encoder_dense_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp=assignvariableop_10_credict_fraud_detect_encoder_dense_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp?assignvariableop_11_credict_fraud_detect_encoder_dense_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp=assignvariableop_12_credict_fraud_detect_encoder_dense_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp?assignvariableop_13_credict_fraud_detect_encoder_dense_4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp=assignvariableop_14_credict_fraud_detect_encoder_dense_4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp?assignvariableop_15_credict_fraud_detect_decoder_dense_5_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp=assignvariableop_16_credict_fraud_detect_decoder_dense_5_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp?assignvariableop_17_credict_fraud_detect_decoder_dense_6_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp=assignvariableop_18_credict_fraud_detect_decoder_dense_6_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp?assignvariableop_19_credict_fraud_detect_decoder_dense_7_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp=assignvariableop_20_credict_fraud_detect_decoder_dense_7_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp?assignvariableop_21_credict_fraud_detect_decoder_dense_8_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp=assignvariableop_22_credict_fraud_detect_decoder_dense_8_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp7assignvariableop_23_credict_fraud_detect_dense_9_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp5assignvariableop_24_credict_fraud_detect_dense_9_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_true_positivesIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp"assignvariableop_30_true_negativesIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_false_positivesIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_false_negativesIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp$assignvariableop_33_true_positives_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp%assignvariableop_34_false_negatives_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpDassignvariableop_35_adam_credict_fraud_detect_encoder_dense_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpBassignvariableop_36_adam_credict_fraud_detect_encoder_dense_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpFassignvariableop_37_adam_credict_fraud_detect_encoder_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpDassignvariableop_38_adam_credict_fraud_detect_encoder_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpFassignvariableop_39_adam_credict_fraud_detect_encoder_dense_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpDassignvariableop_40_adam_credict_fraud_detect_encoder_dense_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpFassignvariableop_41_adam_credict_fraud_detect_encoder_dense_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpDassignvariableop_42_adam_credict_fraud_detect_encoder_dense_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpFassignvariableop_43_adam_credict_fraud_detect_encoder_dense_4_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpDassignvariableop_44_adam_credict_fraud_detect_encoder_dense_4_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpFassignvariableop_45_adam_credict_fraud_detect_decoder_dense_5_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpDassignvariableop_46_adam_credict_fraud_detect_decoder_dense_5_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpFassignvariableop_47_adam_credict_fraud_detect_decoder_dense_6_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpDassignvariableop_48_adam_credict_fraud_detect_decoder_dense_6_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpFassignvariableop_49_adam_credict_fraud_detect_decoder_dense_7_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpDassignvariableop_50_adam_credict_fraud_detect_decoder_dense_7_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpFassignvariableop_51_adam_credict_fraud_detect_decoder_dense_8_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpDassignvariableop_52_adam_credict_fraud_detect_decoder_dense_8_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_credict_fraud_detect_dense_9_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp<assignvariableop_54_adam_credict_fraud_detect_dense_9_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpDassignvariableop_55_adam_credict_fraud_detect_encoder_dense_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpBassignvariableop_56_adam_credict_fraud_detect_encoder_dense_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpFassignvariableop_57_adam_credict_fraud_detect_encoder_dense_1_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpDassignvariableop_58_adam_credict_fraud_detect_encoder_dense_1_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpFassignvariableop_59_adam_credict_fraud_detect_encoder_dense_2_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpDassignvariableop_60_adam_credict_fraud_detect_encoder_dense_2_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpFassignvariableop_61_adam_credict_fraud_detect_encoder_dense_3_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpDassignvariableop_62_adam_credict_fraud_detect_encoder_dense_3_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpFassignvariableop_63_adam_credict_fraud_detect_encoder_dense_4_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpDassignvariableop_64_adam_credict_fraud_detect_encoder_dense_4_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpFassignvariableop_65_adam_credict_fraud_detect_decoder_dense_5_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpDassignvariableop_66_adam_credict_fraud_detect_decoder_dense_5_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpFassignvariableop_67_adam_credict_fraud_detect_decoder_dense_6_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpDassignvariableop_68_adam_credict_fraud_detect_decoder_dense_6_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpFassignvariableop_69_adam_credict_fraud_detect_decoder_dense_7_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpDassignvariableop_70_adam_credict_fraud_detect_decoder_dense_7_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpFassignvariableop_71_adam_credict_fraud_detect_decoder_dense_8_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpDassignvariableop_72_adam_credict_fraud_detect_decoder_dense_8_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp>assignvariableop_73_adam_credict_fraud_detect_dense_9_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp<assignvariableop_74_adam_credict_fraud_detect_dense_9_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75?
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
B__inference_model_layer_call_and_return_conditional_losses_3842046

inputs0
credict_fraud_detect_3842004:
??+
credict_fraud_detect_3842006:	?/
credict_fraud_detect_3842008:	?@*
credict_fraud_detect_3842010:@.
credict_fraud_detect_3842012:@ *
credict_fraud_detect_3842014: .
credict_fraud_detect_3842016: *
credict_fraud_detect_3842018:.
credict_fraud_detect_3842020:*
credict_fraud_detect_3842022:.
credict_fraud_detect_3842024:*
credict_fraud_detect_3842026:.
credict_fraud_detect_3842028: *
credict_fraud_detect_3842030: .
credict_fraud_detect_3842032: @*
credict_fraud_detect_3842034:@/
credict_fraud_detect_3842036:	@?+
credict_fraud_detect_3842038:	?/
credict_fraud_detect_3842040:	?*
credict_fraud_detect_3842042:
identity??,credict_fraud_detect/StatefulPartitionedCall?
,credict_fraud_detect/StatefulPartitionedCallStatefulPartitionedCallinputscredict_fraud_detect_3842004credict_fraud_detect_3842006credict_fraud_detect_3842008credict_fraud_detect_3842010credict_fraud_detect_3842012credict_fraud_detect_3842014credict_fraud_detect_3842016credict_fraud_detect_3842018credict_fraud_detect_3842020credict_fraud_detect_3842022credict_fraud_detect_3842024credict_fraud_detect_3842026credict_fraud_detect_3842028credict_fraud_detect_3842030credict_fraud_detect_3842032credict_fraud_detect_3842034credict_fraud_detect_3842036credict_fraud_detect_3842038credict_fraud_detect_3842040credict_fraud_detect_3842042* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_38418932.
,credict_fraud_detect/StatefulPartitionedCall?
IdentityIdentity5credict_fraud_detect/StatefulPartitionedCall:output:0-^credict_fraud_detect/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2\
,credict_fraud_detect/StatefulPartitionedCall,credict_fraud_detect/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3842457

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:	@?

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_38420462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
D__inference_encoder_layer_call_and_return_conditional_losses_3841807

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/BiasAdd?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/BiasAdd:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/BiasAdd:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAdd?
IdentityIdentitydense_4/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3842269
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:	@?

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_38421812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
%__inference_signature_wrapper_3842412
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:	@?

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_38417692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?#
?
D__inference_decoder_layer_call_and_return_conditional_losses_3842740

inputs8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: @5
'dense_7_biasadd_readvariableop_resource:@9
&dense_8_matmul_readvariableop_resource:	@?6
'dense_8_biasadd_readvariableop_resource:	?
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_5/BiasAdd:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_6/BiasAdd?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/BiasAdd:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAdd?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldense_7/BiasAdd:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_8/BiasAdd?
IdentityIdentitydense_8/BiasAdd:output:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
١
?*
 __inference__traced_save_3843008
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopH
Dsavev2_credict_fraud_detect_encoder_dense_kernel_read_readvariableopF
Bsavev2_credict_fraud_detect_encoder_dense_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_encoder_dense_1_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_encoder_dense_1_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_encoder_dense_2_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_encoder_dense_2_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_encoder_dense_3_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_encoder_dense_3_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_encoder_dense_4_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_encoder_dense_4_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_decoder_dense_5_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_decoder_dense_5_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_decoder_dense_6_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_decoder_dense_6_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_decoder_dense_7_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_decoder_dense_7_bias_read_readvariableopJ
Fsavev2_credict_fraud_detect_decoder_dense_8_kernel_read_readvariableopH
Dsavev2_credict_fraud_detect_decoder_dense_8_bias_read_readvariableopB
>savev2_credict_fraud_detect_dense_9_kernel_read_readvariableop@
<savev2_credict_fraud_detect_dense_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_kernel_m_read_readvariableopM
Isavev2_adam_credict_fraud_detect_encoder_dense_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_1_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_1_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_2_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_2_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_3_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_3_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_4_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_4_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_5_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_5_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_6_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_6_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_7_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_7_bias_m_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_8_kernel_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_8_bias_m_read_readvariableopI
Esavev2_adam_credict_fraud_detect_dense_9_kernel_m_read_readvariableopG
Csavev2_adam_credict_fraud_detect_dense_9_bias_m_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_kernel_v_read_readvariableopM
Isavev2_adam_credict_fraud_detect_encoder_dense_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_1_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_1_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_2_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_2_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_3_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_3_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_encoder_dense_4_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_encoder_dense_4_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_5_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_5_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_6_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_6_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_7_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_7_bias_v_read_readvariableopQ
Msavev2_adam_credict_fraud_detect_decoder_dense_8_kernel_v_read_readvariableopO
Ksavev2_adam_credict_fraud_detect_decoder_dense_8_bias_v_read_readvariableopI
Esavev2_adam_credict_fraud_detect_dense_9_kernel_v_read_readvariableopG
Csavev2_adam_credict_fraud_detect_dense_9_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?"
value?"B?"LB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopDsavev2_credict_fraud_detect_encoder_dense_kernel_read_readvariableopBsavev2_credict_fraud_detect_encoder_dense_bias_read_readvariableopFsavev2_credict_fraud_detect_encoder_dense_1_kernel_read_readvariableopDsavev2_credict_fraud_detect_encoder_dense_1_bias_read_readvariableopFsavev2_credict_fraud_detect_encoder_dense_2_kernel_read_readvariableopDsavev2_credict_fraud_detect_encoder_dense_2_bias_read_readvariableopFsavev2_credict_fraud_detect_encoder_dense_3_kernel_read_readvariableopDsavev2_credict_fraud_detect_encoder_dense_3_bias_read_readvariableopFsavev2_credict_fraud_detect_encoder_dense_4_kernel_read_readvariableopDsavev2_credict_fraud_detect_encoder_dense_4_bias_read_readvariableopFsavev2_credict_fraud_detect_decoder_dense_5_kernel_read_readvariableopDsavev2_credict_fraud_detect_decoder_dense_5_bias_read_readvariableopFsavev2_credict_fraud_detect_decoder_dense_6_kernel_read_readvariableopDsavev2_credict_fraud_detect_decoder_dense_6_bias_read_readvariableopFsavev2_credict_fraud_detect_decoder_dense_7_kernel_read_readvariableopDsavev2_credict_fraud_detect_decoder_dense_7_bias_read_readvariableopFsavev2_credict_fraud_detect_decoder_dense_8_kernel_read_readvariableopDsavev2_credict_fraud_detect_decoder_dense_8_bias_read_readvariableop>savev2_credict_fraud_detect_dense_9_kernel_read_readvariableop<savev2_credict_fraud_detect_dense_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_kernel_m_read_readvariableopIsavev2_adam_credict_fraud_detect_encoder_dense_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_1_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_1_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_2_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_2_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_3_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_3_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_4_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_4_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_5_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_5_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_6_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_6_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_7_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_7_bias_m_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_8_kernel_m_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_8_bias_m_read_readvariableopEsavev2_adam_credict_fraud_detect_dense_9_kernel_m_read_readvariableopCsavev2_adam_credict_fraud_detect_dense_9_bias_m_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_kernel_v_read_readvariableopIsavev2_adam_credict_fraud_detect_encoder_dense_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_1_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_1_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_2_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_2_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_3_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_3_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_encoder_dense_4_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_encoder_dense_4_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_5_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_5_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_6_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_6_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_7_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_7_bias_v_read_readvariableopMsavev2_adam_credict_fraud_detect_decoder_dense_8_kernel_v_read_readvariableopKsavev2_adam_credict_fraud_detect_decoder_dense_8_bias_v_read_readvariableopEsavev2_adam_credict_fraud_detect_dense_9_kernel_v_read_readvariableopCsavev2_adam_credict_fraud_detect_dense_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :
??:?:	?@:@:@ : : :::::: : : @:@:	@?:?:	?:: : : : :?:?:?:?:::
??:?:	?@:@:@ : : :::::: : : @:@:	@?:?:	?::
??:?:	?@:@:@ : : :::::: : : @:@:	@?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 	

_output_shapes
:@:$
 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:! 

_output_shapes	
:?:!!

_output_shapes	
:?: "

_output_shapes
:: #

_output_shapes
::&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?@: '

_output_shapes
:@:$( 

_output_shapes

:@ : )

_output_shapes
: :$* 

_output_shapes

: : +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

: : 1

_output_shapes
: :$2 

_output_shapes

: @: 3

_output_shapes
:@:%4!

_output_shapes
:	@?:!5

_output_shapes	
:?:%6!

_output_shapes
:	?: 7

_output_shapes
::&8"
 
_output_shapes
:
??:!9

_output_shapes	
:?:%:!

_output_shapes
:	?@: ;

_output_shapes
:@:$< 

_output_shapes

:@ : =

_output_shapes
: :$> 

_output_shapes

: : ?

_output_shapes
::$@ 

_output_shapes

:: A

_output_shapes
::$B 

_output_shapes

:: C

_output_shapes
::$D 

_output_shapes

: : E

_output_shapes
: :$F 

_output_shapes

: @: G

_output_shapes
:@:%H!

_output_shapes
:	@?:!I

_output_shapes	
:?:%J!

_output_shapes
:	?: K

_output_shapes
::L

_output_shapes
: 
??
?
"__inference__wrapped_model_3841769
input_1[
Gmodel_credict_fraud_detect_encoder_dense_matmul_readvariableop_resource:
??W
Hmodel_credict_fraud_detect_encoder_dense_biasadd_readvariableop_resource:	?\
Imodel_credict_fraud_detect_encoder_dense_1_matmul_readvariableop_resource:	?@X
Jmodel_credict_fraud_detect_encoder_dense_1_biasadd_readvariableop_resource:@[
Imodel_credict_fraud_detect_encoder_dense_2_matmul_readvariableop_resource:@ X
Jmodel_credict_fraud_detect_encoder_dense_2_biasadd_readvariableop_resource: [
Imodel_credict_fraud_detect_encoder_dense_3_matmul_readvariableop_resource: X
Jmodel_credict_fraud_detect_encoder_dense_3_biasadd_readvariableop_resource:[
Imodel_credict_fraud_detect_encoder_dense_4_matmul_readvariableop_resource:X
Jmodel_credict_fraud_detect_encoder_dense_4_biasadd_readvariableop_resource:[
Imodel_credict_fraud_detect_decoder_dense_5_matmul_readvariableop_resource:X
Jmodel_credict_fraud_detect_decoder_dense_5_biasadd_readvariableop_resource:[
Imodel_credict_fraud_detect_decoder_dense_6_matmul_readvariableop_resource: X
Jmodel_credict_fraud_detect_decoder_dense_6_biasadd_readvariableop_resource: [
Imodel_credict_fraud_detect_decoder_dense_7_matmul_readvariableop_resource: @X
Jmodel_credict_fraud_detect_decoder_dense_7_biasadd_readvariableop_resource:@\
Imodel_credict_fraud_detect_decoder_dense_8_matmul_readvariableop_resource:	@?Y
Jmodel_credict_fraud_detect_decoder_dense_8_biasadd_readvariableop_resource:	?T
Amodel_credict_fraud_detect_dense_9_matmul_readvariableop_resource:	?P
Bmodel_credict_fraud_detect_dense_9_biasadd_readvariableop_resource:
identity??Amodel/credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp?Amodel/credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp?Amodel/credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp?Amodel/credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp?9model/credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp?8model/credict_fraud_detect/dense_9/MatMul/ReadVariableOp??model/credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?>model/credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp?Amodel/credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp?Amodel/credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp?Amodel/credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp?Amodel/credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp?@model/credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp?
>model/credict_fraud_detect/encoder/dense/MatMul/ReadVariableOpReadVariableOpGmodel_credict_fraud_detect_encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02@
>model/credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp?
/model/credict_fraud_detect/encoder/dense/MatMulMatMulinput_1Fmodel/credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/model/credict_fraud_detect/encoder/dense/MatMul?
?model/credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOpReadVariableOpHmodel_credict_fraud_detect_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02A
?model/credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?
0model/credict_fraud_detect/encoder/dense/BiasAddBiasAdd9model/credict_fraud_detect/encoder/dense/MatMul:product:0Gmodel/credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0model/credict_fraud_detect/encoder/dense/BiasAdd?
@model/credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02B
@model/credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp?
1model/credict_fraud_detect/encoder/dense_1/MatMulMatMul9model/credict_fraud_detect/encoder/dense/BiasAdd:output:0Hmodel/credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@23
1model/credict_fraud_detect/encoder/dense_1/MatMul?
Amodel/credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amodel/credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/encoder/dense_1/BiasAddBiasAdd;model/credict_fraud_detect/encoder/dense_1/MatMul:product:0Imodel/credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@24
2model/credict_fraud_detect/encoder/dense_1/BiasAdd?
@model/credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_encoder_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02B
@model/credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp?
1model/credict_fraud_detect/encoder/dense_2/MatMulMatMul;model/credict_fraud_detect/encoder/dense_1/BiasAdd:output:0Hmodel/credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 23
1model/credict_fraud_detect/encoder/dense_2/MatMul?
Amodel/credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amodel/credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/encoder/dense_2/BiasAddBiasAdd;model/credict_fraud_detect/encoder/dense_2/MatMul:product:0Imodel/credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 24
2model/credict_fraud_detect/encoder/dense_2/BiasAdd?
@model/credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_encoder_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02B
@model/credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp?
1model/credict_fraud_detect/encoder/dense_3/MatMulMatMul;model/credict_fraud_detect/encoder/dense_2/BiasAdd:output:0Hmodel/credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????23
1model/credict_fraud_detect/encoder/dense_3/MatMul?
Amodel/credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Amodel/credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/encoder/dense_3/BiasAddBiasAdd;model/credict_fraud_detect/encoder/dense_3/MatMul:product:0Imodel/credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2model/credict_fraud_detect/encoder/dense_3/BiasAdd?
@model/credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_encoder_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02B
@model/credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp?
1model/credict_fraud_detect/encoder/dense_4/MatMulMatMul;model/credict_fraud_detect/encoder/dense_3/BiasAdd:output:0Hmodel/credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????23
1model/credict_fraud_detect/encoder/dense_4/MatMul?
Amodel/credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Amodel/credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/encoder/dense_4/BiasAddBiasAdd;model/credict_fraud_detect/encoder/dense_4/MatMul:product:0Imodel/credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2model/credict_fraud_detect/encoder/dense_4/BiasAdd?
@model/credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_decoder_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02B
@model/credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp?
1model/credict_fraud_detect/decoder/dense_5/MatMulMatMul;model/credict_fraud_detect/encoder/dense_4/BiasAdd:output:0Hmodel/credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????23
1model/credict_fraud_detect/decoder/dense_5/MatMul?
Amodel/credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_decoder_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Amodel/credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/decoder/dense_5/BiasAddBiasAdd;model/credict_fraud_detect/decoder/dense_5/MatMul:product:0Imodel/credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????24
2model/credict_fraud_detect/decoder/dense_5/BiasAdd?
@model/credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_decoder_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02B
@model/credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp?
1model/credict_fraud_detect/decoder/dense_6/MatMulMatMul;model/credict_fraud_detect/decoder/dense_5/BiasAdd:output:0Hmodel/credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 23
1model/credict_fraud_detect/decoder/dense_6/MatMul?
Amodel/credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_decoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amodel/credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/decoder/dense_6/BiasAddBiasAdd;model/credict_fraud_detect/decoder/dense_6/MatMul:product:0Imodel/credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 24
2model/credict_fraud_detect/decoder/dense_6/BiasAdd?
@model/credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_decoder_dense_7_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02B
@model/credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp?
1model/credict_fraud_detect/decoder/dense_7/MatMulMatMul;model/credict_fraud_detect/decoder/dense_6/BiasAdd:output:0Hmodel/credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@23
1model/credict_fraud_detect/decoder/dense_7/MatMul?
Amodel/credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_decoder_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amodel/credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/decoder/dense_7/BiasAddBiasAdd;model/credict_fraud_detect/decoder/dense_7/MatMul:product:0Imodel/credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@24
2model/credict_fraud_detect/decoder/dense_7/BiasAdd?
@model/credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOpReadVariableOpImodel_credict_fraud_detect_decoder_dense_8_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02B
@model/credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp?
1model/credict_fraud_detect/decoder/dense_8/MatMulMatMul;model/credict_fraud_detect/decoder/dense_7/BiasAdd:output:0Hmodel/credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1model/credict_fraud_detect/decoder/dense_8/MatMul?
Amodel/credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOpReadVariableOpJmodel_credict_fraud_detect_decoder_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Amodel/credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp?
2model/credict_fraud_detect/decoder/dense_8/BiasAddBiasAdd;model/credict_fraud_detect/decoder/dense_8/MatMul:product:0Imodel/credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2model/credict_fraud_detect/decoder/dense_8/BiasAdd?
8model/credict_fraud_detect/dense_9/MatMul/ReadVariableOpReadVariableOpAmodel_credict_fraud_detect_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02:
8model/credict_fraud_detect/dense_9/MatMul/ReadVariableOp?
)model/credict_fraud_detect/dense_9/MatMulMatMul;model/credict_fraud_detect/decoder/dense_8/BiasAdd:output:0@model/credict_fraud_detect/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)model/credict_fraud_detect/dense_9/MatMul?
9model/credict_fraud_detect/dense_9/BiasAdd/ReadVariableOpReadVariableOpBmodel_credict_fraud_detect_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9model/credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp?
*model/credict_fraud_detect/dense_9/BiasAddBiasAdd3model/credict_fraud_detect/dense_9/MatMul:product:0Amodel/credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*model/credict_fraud_detect/dense_9/BiasAdd?
*model/credict_fraud_detect/dense_9/SoftmaxSoftmax3model/credict_fraud_detect/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2,
*model/credict_fraud_detect/dense_9/Softmax?
IdentityIdentity4model/credict_fraud_detect/dense_9/Softmax:softmax:0B^model/credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOpB^model/credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOpB^model/credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOpB^model/credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp:^model/credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp9^model/credict_fraud_detect/dense_9/MatMul/ReadVariableOp@^model/credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?^model/credict_fraud_detect/encoder/dense/MatMul/ReadVariableOpB^model/credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOpB^model/credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOpB^model/credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOpB^model/credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOpA^model/credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2?
Amodel/credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/decoder/dense_5/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp@model/credict_fraud_detect/decoder/dense_5/MatMul/ReadVariableOp2?
Amodel/credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/decoder/dense_6/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp@model/credict_fraud_detect/decoder/dense_6/MatMul/ReadVariableOp2?
Amodel/credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/decoder/dense_7/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp@model/credict_fraud_detect/decoder/dense_7/MatMul/ReadVariableOp2?
Amodel/credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/decoder/dense_8/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp@model/credict_fraud_detect/decoder/dense_8/MatMul/ReadVariableOp2v
9model/credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp9model/credict_fraud_detect/dense_9/BiasAdd/ReadVariableOp2t
8model/credict_fraud_detect/dense_9/MatMul/ReadVariableOp8model/credict_fraud_detect/dense_9/MatMul/ReadVariableOp2?
?model/credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp?model/credict_fraud_detect/encoder/dense/BiasAdd/ReadVariableOp2?
>model/credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp>model/credict_fraud_detect/encoder/dense/MatMul/ReadVariableOp2?
Amodel/credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/encoder/dense_1/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp@model/credict_fraud_detect/encoder/dense_1/MatMul/ReadVariableOp2?
Amodel/credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/encoder/dense_2/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp@model/credict_fraud_detect/encoder/dense_2/MatMul/ReadVariableOp2?
Amodel/credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/encoder/dense_3/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp@model/credict_fraud_detect/encoder/dense_3/MatMul/ReadVariableOp2?
Amodel/credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOpAmodel/credict_fraud_detect/encoder/dense_4/BiasAdd/ReadVariableOp2?
@model/credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp@model/credict_fraud_detect/encoder/dense_4/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
D__inference_dense_9_layer_call_and_return_conditional_losses_3842760

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3842089
input_1
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:	@?

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_38420462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
B__inference_model_layer_call_and_return_conditional_losses_3842181

inputs0
credict_fraud_detect_3842139:
??+
credict_fraud_detect_3842141:	?/
credict_fraud_detect_3842143:	?@*
credict_fraud_detect_3842145:@.
credict_fraud_detect_3842147:@ *
credict_fraud_detect_3842149: .
credict_fraud_detect_3842151: *
credict_fraud_detect_3842153:.
credict_fraud_detect_3842155:*
credict_fraud_detect_3842157:.
credict_fraud_detect_3842159:*
credict_fraud_detect_3842161:.
credict_fraud_detect_3842163: *
credict_fraud_detect_3842165: .
credict_fraud_detect_3842167: @*
credict_fraud_detect_3842169:@/
credict_fraud_detect_3842171:	@?+
credict_fraud_detect_3842173:	?/
credict_fraud_detect_3842175:	?*
credict_fraud_detect_3842177:
identity??,credict_fraud_detect/StatefulPartitionedCall?
,credict_fraud_detect/StatefulPartitionedCallStatefulPartitionedCallinputscredict_fraud_detect_3842139credict_fraud_detect_3842141credict_fraud_detect_3842143credict_fraud_detect_3842145credict_fraud_detect_3842147credict_fraud_detect_3842149credict_fraud_detect_3842151credict_fraud_detect_3842153credict_fraud_detect_3842155credict_fraud_detect_3842157credict_fraud_detect_3842159credict_fraud_detect_3842161credict_fraud_detect_3842163credict_fraud_detect_3842165credict_fraud_detect_3842167credict_fraud_detect_3842169credict_fraud_detect_3842171credict_fraud_detect_3842173credict_fraud_detect_3842175credict_fraud_detect_3842177* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_38418932.
,credict_fraud_detect/StatefulPartitionedCall?
IdentityIdentity5credict_fraud_detect/StatefulPartitionedCall:output:0-^credict_fraud_detect/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2\
,credict_fraud_detect/StatefulPartitionedCall,credict_fraud_detect/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense_9_layer_call_fn_3842749

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_38418862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_3842712

inputs
unknown:
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:	@?
	unknown_6:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_38418572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_9_layer_call_and_return_conditional_losses_3841886

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_encoder_layer_call_fn_3842657

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_38418072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_model_layer_call_and_return_conditional_losses_3842314
input_10
credict_fraud_detect_3842272:
??+
credict_fraud_detect_3842274:	?/
credict_fraud_detect_3842276:	?@*
credict_fraud_detect_3842278:@.
credict_fraud_detect_3842280:@ *
credict_fraud_detect_3842282: .
credict_fraud_detect_3842284: *
credict_fraud_detect_3842286:.
credict_fraud_detect_3842288:*
credict_fraud_detect_3842290:.
credict_fraud_detect_3842292:*
credict_fraud_detect_3842294:.
credict_fraud_detect_3842296: *
credict_fraud_detect_3842298: .
credict_fraud_detect_3842300: @*
credict_fraud_detect_3842302:@/
credict_fraud_detect_3842304:	@?+
credict_fraud_detect_3842306:	?/
credict_fraud_detect_3842308:	?*
credict_fraud_detect_3842310:
identity??,credict_fraud_detect/StatefulPartitionedCall?
,credict_fraud_detect/StatefulPartitionedCallStatefulPartitionedCallinput_1credict_fraud_detect_3842272credict_fraud_detect_3842274credict_fraud_detect_3842276credict_fraud_detect_3842278credict_fraud_detect_3842280credict_fraud_detect_3842282credict_fraud_detect_3842284credict_fraud_detect_3842286credict_fraud_detect_3842288credict_fraud_detect_3842290credict_fraud_detect_3842292credict_fraud_detect_3842294credict_fraud_detect_3842296credict_fraud_detect_3842298credict_fraud_detect_3842300credict_fraud_detect_3842302credict_fraud_detect_3842304credict_fraud_detect_3842306credict_fraud_detect_3842308credict_fraud_detect_3842310* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_38418932.
,credict_fraud_detect/StatefulPartitionedCall?
IdentityIdentity5credict_fraud_detect/StatefulPartitionedCall:output:0-^credict_fraud_detect/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2\
,credict_fraud_detect/StatefulPartitionedCall,credict_fraud_detect/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?+
?
D__inference_encoder_layer_call_and_return_conditional_losses_3842691

inputs8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAdd?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/BiasAdd:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/BiasAdd?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/BiasAdd:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/BiasAdd:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAdd?
IdentityIdentitydense_4/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_3842502

inputs
unknown:
??
	unknown_0:	?
	unknown_1:	?@
	unknown_2:@
	unknown_3:@ 
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11: 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:	@?

unknown_16:	?

unknown_17:	?

unknown_18:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_38421812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_3841893
input_1#
encoder_3841808:
??
encoder_3841810:	?"
encoder_3841812:	?@
encoder_3841814:@!
encoder_3841816:@ 
encoder_3841818: !
encoder_3841820: 
encoder_3841822:!
encoder_3841824:
encoder_3841826:!
decoder_3841858:
decoder_3841860:!
decoder_3841862: 
decoder_3841864: !
decoder_3841866: @
decoder_3841868:@"
decoder_3841870:	@?
decoder_3841872:	?"
dense_9_3841887:	?
dense_9_3841889:
identity??decoder/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_3841808encoder_3841810encoder_3841812encoder_3841814encoder_3841816encoder_3841818encoder_3841820encoder_3841822encoder_3841824encoder_3841826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_38418072!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_3841858decoder_3841860decoder_3841862decoder_3841864decoder_3841866decoder_3841868decoder_3841870decoder_3841872*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_38418572!
decoder/StatefulPartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(decoder/StatefulPartitionedCall:output:0dense_9_3841887dense_9_3841889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_38418862!
dense_9/StatefulPartitionedCall?
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:??????????: : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????H
credict_fraud_detect0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?3
layer-0
layer_with_weights-0
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?1
_tf_keras_network?1{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 203]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "CredictFraudDetect", "config": {"layer was saved without config": true}, "name": "credict_fraud_detect", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["credict_fraud_detect", 0, 0]]}, "shared_object_id": 1, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 203]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 203]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 203]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 3}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 4}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}, "shared_object_id": 5}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": 1}, "shared_object_id": 6}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 1.000000082740371e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 203]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 203]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
	encoder

decoder
	dense
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"name": "credict_fraud_detect", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "CredictFraudDetect", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 203]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "CredictFraudDetect"}}
?
iter

beta_1

beta_2
	decay
learning_ratem?m?m?m?m?m?m?m?m?m?m? m?!m?"m?#m?$m?%m?&m?'m?(m?v?v?v?v?v?v?v?v?v?v?v? v?!v?"v?#v?$v?%v?&v?'v?(v?"
	optimizer
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
)metrics

*layers
+layer_regularization_losses
,layer_metrics
	variables
-non_trainable_variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
.	dense_128
/dense_64
0dense_32
1dense_8
2dense_4
3regularization_losses
4	variables
5trainable_variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Encoder", "config": {"name": "encoder", "trainable": true, "dtype": "float32"}, "shared_object_id": 7}
?
7dense_8
8dense_32
9dense_64
:	dense_128
;regularization_losses
<	variables
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Decoder", "config": {"name": "decoder", "trainable": true, "dtype": "float32"}, "shared_object_id": 8}
?

'kernel
(bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15
%16
&17
'18
(19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cmetrics

Dlayers
Elayer_regularization_losses
Flayer_metrics
	variables
Gnon_trainable_variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
=:;
??2)credict_fraud_detect/encoder/dense/kernel
6:4?2'credict_fraud_detect/encoder/dense/bias
>:<	?@2+credict_fraud_detect/encoder/dense_1/kernel
7:5@2)credict_fraud_detect/encoder/dense_1/bias
=:;@ 2+credict_fraud_detect/encoder/dense_2/kernel
7:5 2)credict_fraud_detect/encoder/dense_2/bias
=:; 2+credict_fraud_detect/encoder/dense_3/kernel
7:52)credict_fraud_detect/encoder/dense_3/bias
=:;2+credict_fraud_detect/encoder/dense_4/kernel
7:52)credict_fraud_detect/encoder/dense_4/bias
=:;2+credict_fraud_detect/decoder/dense_5/kernel
7:52)credict_fraud_detect/decoder/dense_5/bias
=:; 2+credict_fraud_detect/decoder/dense_6/kernel
7:5 2)credict_fraud_detect/decoder/dense_6/bias
=:; @2+credict_fraud_detect/decoder/dense_7/kernel
7:5@2)credict_fraud_detect/decoder/dense_7/bias
>:<	@?2+credict_fraud_detect/decoder/dense_8/kernel
8:6?2)credict_fraud_detect/decoder/dense_8/bias
6:4	?2#credict_fraud_detect/dense_9/kernel
/:-2!credict_fraud_detect/dense_9/bias
<
H0
I1
J2
K3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

kernel
bias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 203}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 203]}}
?

kernel
bias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

kernel
bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

kernel
bias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

kernel
bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
?
3regularization_losses

`layers
alayer_regularization_losses
blayer_metrics
4	variables
cnon_trainable_variables
5trainable_variables
dmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

kernel
 bias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?

!kernel
"bias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?

#kernel
$bias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

%kernel
&bias
qregularization_losses
r	variables
strainable_variables
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
X
0
 1
!2
"3
#4
$5
%6
&7"
trackable_list_wrapper
X
0
 1
!2
"3
#4
$5
%6
&7"
trackable_list_wrapper
?
;regularization_losses

ulayers
vlayer_regularization_losses
wlayer_metrics
<	variables
xnon_trainable_variables
=trainable_variables
ymetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
?regularization_losses

zlayers
{layer_regularization_losses
|layer_metrics
@	variables
}non_trainable_variables
Atrainable_variables
~metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
	0

1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 4}
?"
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"?!
_tf_keras_metric?!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}, "shared_object_id": 5}
?
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": 1}, "shared_object_id": 6}
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Lregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
M	variables
?non_trainable_variables
Ntrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Pregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Q	variables
?non_trainable_variables
Rtrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Tregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
U	variables
?non_trainable_variables
Vtrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Xregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Y	variables
?non_trainable_variables
Ztrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
\regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
]	variables
?non_trainable_variables
^trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C
.0
/1
02
13
24"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
eregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
f	variables
?non_trainable_variables
gtrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
iregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
j	variables
?non_trainable_variables
ktrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
mregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
n	variables
?non_trainable_variables
otrainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
qregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
r	variables
?non_trainable_variables
strainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
<
70
81
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
/
0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
B:@
??20Adam/credict_fraud_detect/encoder/dense/kernel/m
;:9?2.Adam/credict_fraud_detect/encoder/dense/bias/m
C:A	?@22Adam/credict_fraud_detect/encoder/dense_1/kernel/m
<::@20Adam/credict_fraud_detect/encoder/dense_1/bias/m
B:@@ 22Adam/credict_fraud_detect/encoder/dense_2/kernel/m
<:: 20Adam/credict_fraud_detect/encoder/dense_2/bias/m
B:@ 22Adam/credict_fraud_detect/encoder/dense_3/kernel/m
<::20Adam/credict_fraud_detect/encoder/dense_3/bias/m
B:@22Adam/credict_fraud_detect/encoder/dense_4/kernel/m
<::20Adam/credict_fraud_detect/encoder/dense_4/bias/m
B:@22Adam/credict_fraud_detect/decoder/dense_5/kernel/m
<::20Adam/credict_fraud_detect/decoder/dense_5/bias/m
B:@ 22Adam/credict_fraud_detect/decoder/dense_6/kernel/m
<:: 20Adam/credict_fraud_detect/decoder/dense_6/bias/m
B:@ @22Adam/credict_fraud_detect/decoder/dense_7/kernel/m
<::@20Adam/credict_fraud_detect/decoder/dense_7/bias/m
C:A	@?22Adam/credict_fraud_detect/decoder/dense_8/kernel/m
=:;?20Adam/credict_fraud_detect/decoder/dense_8/bias/m
;:9	?2*Adam/credict_fraud_detect/dense_9/kernel/m
4:22(Adam/credict_fraud_detect/dense_9/bias/m
B:@
??20Adam/credict_fraud_detect/encoder/dense/kernel/v
;:9?2.Adam/credict_fraud_detect/encoder/dense/bias/v
C:A	?@22Adam/credict_fraud_detect/encoder/dense_1/kernel/v
<::@20Adam/credict_fraud_detect/encoder/dense_1/bias/v
B:@@ 22Adam/credict_fraud_detect/encoder/dense_2/kernel/v
<:: 20Adam/credict_fraud_detect/encoder/dense_2/bias/v
B:@ 22Adam/credict_fraud_detect/encoder/dense_3/kernel/v
<::20Adam/credict_fraud_detect/encoder/dense_3/bias/v
B:@22Adam/credict_fraud_detect/encoder/dense_4/kernel/v
<::20Adam/credict_fraud_detect/encoder/dense_4/bias/v
B:@22Adam/credict_fraud_detect/decoder/dense_5/kernel/v
<::20Adam/credict_fraud_detect/decoder/dense_5/bias/v
B:@ 22Adam/credict_fraud_detect/decoder/dense_6/kernel/v
<:: 20Adam/credict_fraud_detect/decoder/dense_6/bias/v
B:@ @22Adam/credict_fraud_detect/decoder/dense_7/kernel/v
<::@20Adam/credict_fraud_detect/decoder/dense_7/bias/v
C:A	@?22Adam/credict_fraud_detect/decoder/dense_8/kernel/v
=:;?20Adam/credict_fraud_detect/decoder/dense_8/bias/v
;:9	?2*Adam/credict_fraud_detect/dense_9/kernel/v
4:22(Adam/credict_fraud_detect/dense_9/bias/v
?2?
'__inference_model_layer_call_fn_3842089
'__inference_model_layer_call_fn_3842457
'__inference_model_layer_call_fn_3842502
'__inference_model_layer_call_fn_3842269?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_3841769?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
B__inference_model_layer_call_and_return_conditional_losses_3842567
B__inference_model_layer_call_and_return_conditional_losses_3842632
B__inference_model_layer_call_and_return_conditional_losses_3842314
B__inference_model_layer_call_and_return_conditional_losses_3842359?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_credict_fraud_detect_layer_call_fn_3841939?
???
FullArgSpec
args?
jself
jinp
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_3841893?
???
FullArgSpec
args?
jself
jinp
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?B?
%__inference_signature_wrapper_3842412input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_encoder_layer_call_fn_3842657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_encoder_layer_call_and_return_conditional_losses_3842691?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_decoder_layer_call_fn_3842712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_decoder_layer_call_and_return_conditional_losses_3842740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_9_layer_call_fn_3842749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_9_layer_call_and_return_conditional_losses_3842760?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_3841769? !"#$%&'(1?.
'?$
"?
input_1??????????
? "K?H
F
credict_fraud_detect.?+
credict_fraud_detect??????????
Q__inference_credict_fraud_detect_layer_call_and_return_conditional_losses_3841893p !"#$%&'(1?.
'?$
"?
input_1??????????
? "%?"
?
0?????????
? ?
6__inference_credict_fraud_detect_layer_call_fn_3841939c !"#$%&'(1?.
'?$
"?
input_1??????????
? "???????????
D__inference_decoder_layer_call_and_return_conditional_losses_3842740c !"#$%&/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_decoder_layer_call_fn_3842712V !"#$%&/?,
%?"
 ?
inputs?????????
? "????????????
D__inference_dense_9_layer_call_and_return_conditional_losses_3842760]'(0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_9_layer_call_fn_3842749P'(0?-
&?#
!?
inputs??????????
? "???????????
D__inference_encoder_layer_call_and_return_conditional_losses_3842691e
0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
)__inference_encoder_layer_call_fn_3842657X
0?-
&?#
!?
inputs??????????
? "???????????
B__inference_model_layer_call_and_return_conditional_losses_3842314x !"#$%&'(9?6
/?,
"?
input_1??????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_3842359x !"#$%&'(9?6
/?,
"?
input_1??????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_3842567w !"#$%&'(8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_3842632w !"#$%&'(8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_layer_call_fn_3842089k !"#$%&'(9?6
/?,
"?
input_1??????????
p 

 
? "???????????
'__inference_model_layer_call_fn_3842269k !"#$%&'(9?6
/?,
"?
input_1??????????
p

 
? "???????????
'__inference_model_layer_call_fn_3842457j !"#$%&'(8?5
.?+
!?
inputs??????????
p 

 
? "???????????
'__inference_model_layer_call_fn_3842502j !"#$%&'(8?5
.?+
!?
inputs??????????
p

 
? "???????????
%__inference_signature_wrapper_3842412? !"#$%&'(<?9
? 
2?/
-
input_1"?
input_1??????????"K?H
F
credict_fraud_detect.?+
credict_fraud_detect?????????