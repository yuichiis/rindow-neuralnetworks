<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class Embedding extends AbstractLayer
{
    use GenericUtils;
    protected ?int $inputLength;
    protected int $inputDim;
    protected int $outputDim;
    protected mixed $kernelInitializer;
    protected ?string $kernelInitializerName;
    protected ?int $inputDtype=NDArray::int32;

    protected ?NDArray $kernel=null;
    protected NDArray $dKernel;
    //protected $inputs;
    //protected $originalShape;
    //protected $flattenOutputsShape;

    public function __construct(
        object $backend,
        int $inputDim,
        int $outputDim,
        int $input_length=null,
        string|callable $kernel_initializer=null,
        string $name=null,
    )
    {
        // defaults
        $input_length = $input_length ?? null;
        $kernel_initializer = $kernel_initializer ?? 'random_uniform';
        $name = $name ?? null;
        
        parent::__construct($backend);
        $K = $backend;
        if($input_length!=null){
            $this->inputShape = [$input_length];
        }
        $this->inputLength = $input_length;
        $this->inputDim = $inputDim;
        $this->outputDim = $outputDim;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->kernelInitializerName = $this->toStringName($kernel_initializer);
        $this->initName($name,'embedding');
        $this->allocateWeights(1);
    }

    public function build(mixed $variable=null, array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;

        $inputShape = $this->normalizeInputShape($variable);
        if(count($inputShape)!=1) {
            throw new InvalidArgumentException(
                'Unsuppored input shape: ['.implode(',',$inputShape).']');
        }
        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
            } else {
                $this->kernel = $kernelInitializer(
                    [$this->inputDim,$this->outputDim],
                    [$this->inputDim,$this->outputDim]
                );
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->outputShape = array_merge($inputShape,[$this->outputDim]);
        $this->syncWeightVariables();
    }

    public function getParams() : array
    {
        return [$this->kernel];
    }

    public function getGrads() : array
    {
        return [$this->dKernel];
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->kernel = $this->weights[0]->value();
    }

    public function getConfig() : array
    {
        return [
            'inputDim' => $this->inputDim,
            'outputDim' => $this->outputDim,
            'options' => [
                'input_length'=>$this->inputLength,
                'kernel_initializer' => $this->kernelInitializerName,
            ]
        ];
    }

    /**
     * inputs:  [batch,len]
     * kernel:  [inputDim,outputDim] (numClass=inputDim)
     * outputs: [batch,len,outputDim]
     */
    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->originalShape = $inputs->shape();
        $container->inputs = $inputs;

        $inputs = $inputs->reshape([$inputs->size(),1]);
        // gatherND(
        //  params:  [p0=inputDim,  k=outputDim]   <= kernel
        //  indices: [n=batch*len,  indexDepth=1]  <= inputs
        //  outputs: [n=batch*len,  k=outputDim]   <= outputs
        //  batchDims: 0
        //)
        $outputs = $K->gatherND($this->kernel,$inputs);
        $container->flattenOutputsShape = $outputs->shape();
        $shape = $container->originalShape;
        array_push($shape,$this->outputDim);
        return $outputs->reshape($shape);
    }

    /**
     * dOutputs: [batch,len,outputDim]          (m=batch*len, k=outputDim)
     * inputs:   [batch,len]                    (m=batch*len)
     * scatter:  [batch,len,inputDim,outputDim] (m=batch*len, numClass=inputDim, k=outputDim)
     * kernel:   [inputDim,outputDim]
     */
    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        // dKernel[m=1,x[m,n],k] += dOutputs[m=1,n,k];
        // dKernel[x[n],k] += dOutputs[n,k];

        // === ScatterAdd edition ===
        // dKernel[x[batch],k] += dOutputs[batch,k];
        // $dOutputs = $dOutputs->reshape($container->flattenOutputsShape);
        // $K->clear($this->dKernel);
        // $K->scatterAdd($this->dKernel,$container->inputs, $dOutputs);
        //
        // === Scatter and ReduceSum edition ===
        // tmp[m,x[m,n=1],k] = dOutputs[m,n=1,k];
        // dKernel[numClass,k] = reduceSum(tmp[m,numClass,k],axis=0);
        // [batch,len]
        $indicesShape = $container->originalShape;
        array_push($indicesShape,1);
        $inputs = $container->inputs->reshape($indicesShape);
        // scatterND(
        //  indices: [m=batch, n=len, 1]
        //  updates: [m=batch, n=len, k=outputDim]
        //  outputs: [m=batch, p0=inputDim, k=outputDim]
        //  batchDims: 1
        // )
        $shape = array_merge([$container->originalShape[0]],$this->kernel->shape());
        $dKernel = $K->scatterND($inputs, $dOutputs, shape:$shape, batchDims:1);
        $dKernel = $K->sum($dKernel,axis:0,output:$this->dKernel);

        return $container->inputs->reshape($container->originalShape);//dummy
    }
}
