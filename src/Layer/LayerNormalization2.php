<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class LayerNormalization extends AbstractNormalization
{
    public function __construct(
        object $backend,
        int $axis=null,
        float $epsilon=null,
        bool $center=null,
        bool $scale=null,
        string|callable $beta_initializer=null,
        string|callable $gamma_initializer=null,
        string $name=null,
    )
    {
        parent::__construct(
            $backend,
            $axis,
            $epsilon,
            $center,
            $scale,
            $beta_initializer,
            $gamma_initializer,
        );
        // defaults
        $name = $name ?? null;

        $this->initName($name,'layernormalization');
        $this->allocateWeights(2);
    }

    protected function buildNoTrainingMode(array $kernelShape) : void
    {
    }

    public function reverseSyncWeightVariables() : void
    {
        $this->beta = $this->weights[0]->value();
        $this->gamma = $this->weights[1]->value();
    }

    public function getConfig() : array
    {
        return [
            'options' => [
                'axis'=>$this->axis,
                'epsilon'=>$this->epsilon,
                'beta_initializer'=>$this->betaInitializerName,
                'gamma_initializer'=>$this->gammaInitializerName,
            ]
        ];
    }

    public function getParams() : array
    {
        return [$this->beta,$this->gamma];
    }

    public function getGrads() : array
    {
        return [$this->dBeta,$this->dGamma];
    }

    protected function call(NDArray $inputs, bool $training=null) : NDArray
    {
        //echo "============= call ============================\n";
        $K = $this->backend;
        $mo = $this->backend->localMatrixOperator();
        //if($training===null) {
        //    throw new InvalidArgumentException("training option must be true or false.");
        //}
        $container = $this->container();
        // (batch,heads...,feature) => (batch*heads,feature)
        $inputs = $this->transformShape($inputs);

        // normalization
        // xn = (x - mean(x)) / sqrt(mean( (x - mean(x))**2 ) + eps)
        //

        // mean = mean(x)
        // center = x - mean(x)
        $mean = $K->mean($inputs,axis:-1);                          // (batch*heads)
        //echo "mean=".$mo->shapeToString($mean->shape())."\n";
        $center_x = $K->sub($inputs, $mean, trans:true);            // (batch*heads,feature)
        //echo "sub=".$mo->shapeToString($center_x->shape())."\n";

        // variance = mean(square(x - mean), axis=-1)
        $variance = $K->mean($K->square($center_x), axis:-1);       // (batch*heads)
        //echo "variance=".$mo->shapeToString($variance->shape())."\n";

        // std = sqrt(variance+eps)
        // normalized_x = x-mean(x) / std
        $std = $K->sqrt($K->increment($variance, $this->epsilon));  // (batch*heads)
        $norm_x = $K->div($center_x, $std, trans:true);             // (batch*heads,feature)

        $container->xc = $center_x; // (batch*head,feature)
        $container->xn = $norm_x;   // (batch*head,feature)
        $container->std = $std;     // (batch*heads)
        $container->variance = $variance;     // (batch*heads)

        if($this->gamma) {
            $outputs = $K->mul($this->gamma, $norm_x);
        } else {
            $outputs = $norm_x;
        }
        if($this->beta) {
            $outputs = $K->add($outputs, $this->beta);
        }

        $outputs = $this->untransformShape($outputs);
        return $outputs;
    }

    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        //echo "============= differentiate ============================\n";
        $mo = $this->backend->localMatrixOperator();
        $K = $this->backend;
        $dOutputs = $this->transformShape($dOutputs);
        $container = $this->container();
        $center_x = $container->xc; // (batch*head,feature)
        $norm_x = $container->xn;   // (batch*head,feature)
        $std = $container->std;     // (batch*heads)
        $variance = $container->variance; // (batch*heads)

        $tmp = $dOutputs->shape();
        $feature_dim = array_pop($tmp);


        // d_scaled_x = dOutputs    (batch*head,feature)

        // d_normalized_x = d_scaled_x * gamma      // (batch*head,feature)
        if($this->dBeta) {
            // d_beta = sum(d_scaled_x, axis:0)
            $dbeta = $K->sum($dOutputs,axis:0,output:$this->dBeta);
            //$K->copy($dbeta,$this->dBeta);
        }
        if($this->dGamma) {
            // d_gamma = sum(d_scaled_x*normalized_x, axis:0)
            $dgamma = $K->sum($K->mul($container->xn, $dOutputs), axis:0, output:$this->dGamma);
            //$K->copy($dgamma,$this->dGamma);
            // d_normalized_x = d_scaled_x * gamma
            $d_norm_x = $K->mul($this->gamma, $dOutputs);
        } else {
            $dxn = $dOutputs;
        }

        if($container->std===null) {
            throw new LogicException('not initialized for training');
        }
        //echo "d_norm_x=".$mo->toString($d_norm_x,indent:true)."\n";
        //echo "d_norm_x=".$mo->shapeToString($d_norm_x->shape())."\n";
        //echo "std=".$mo->shapeToString($std->shape())."\n";
        //echo "center_x= ".$mo->shapeToString($center_x->shape())."\n";
        //echo "variance= ".$mo->shapeToString($variance->shape())."\n";

        // d_center_x = d_normalized_x / std            // (batch*heads)
        //$d_center_x = $K->div($d_norm_x, $std, trans:true);

        //// d_std = -sum(d_norm_x*center_x / (std**2))   // (batch*heads)
        //$d_std = $K->scale(-1.0, $K->sum(
        //    $K->div($K->mul($d_norm_x, $center_x), $K->square($std),trans:true),
        //    axis:-1));

        // d_variance = sum(                            // (batch*heads)
        //      -d_normalized_x*center_x/2 * (variance + epsilon)**(-1.5)
        // )

        $d_variance = $K->sum($K->mul(
            $K->scale(-0.5, $K->mul($d_norm_x,$center_x)),
            $K->pow($K->increment($variance, $this->epsilon),-1.5),
            trans:true,
        ),axis:-1);
        //echo "d_variance=".$mo->toString($d_variance,indent:true)."\n";
        

        // d_mean  = sum(-d_normalized_x/sqrt(variance + epsilon)) +    // (batch*heads)
        //           d_variance * sum(-2*center_x) / feature_dim
        $d_mean = $K->add(
            $K->sum(
                $K->div(
                    $K->scale(-1,$d_norm_x),
                    $std,
                    trans:true,
                ),
                axis:-1,
            ),
            $K->scale(
                1/$feature_dim,
                $K->mul(
                    $K->sum($K->scale(-2,$center_x), axis:-1),
                    $d_variance,
                ),
            ),
        );
        //echo "d_mean=".$mo->toString($d_mean,indent:true)."\n";
        //echo "d_mean= ".$mo->shapeToString($d_mean->shape())."\n";


        // d_x = d_normalized_x / sqrt(variance + epsilon) +
        //       d_variance * 2 * center_x / feature_dim_f +
        //       d_mean / feature_dim
        //$tmp1 = $K->div($d_norm_x, $K->sqrt($K->increment($variance,$this->epsilon)),trans:true);
        $tmp1 = $K->div($d_norm_x, $std, trans:true);
        //echo "tmp1=".$mo->toString($tmp1,indent:true)."\n";
        $tmp2 = $K->scale(1/$feature_dim,$K->mul($center_x,$K->scale(2,$d_variance),trans:true));
        //echo "tmp2=".$mo->toString($tmp2,indent:true)."\n";

        $dInputs = $K->add(
            $K->add(
                $tmp1,
                $tmp2,
            ),
            $K->scale(1/$feature_dim,$d_mean),
            trans:true,
        );
        //echo "d_x=".$mo->toString($dInputs,indent:true)."\n";

        $dInputs = $this->untransformShape($dInputs);

        return $dInputs;

/*
        mean = np.mean(x, axis=1, keepdims=True)
        variance = np.var(x, axis=1, keepdims=True)
        normalized_x = (x - mean) / np.sqrt(variance + epsilon)
        scaled_x = gamma * normalized_x + beta
    
        # Gradient calculation
        d_scaled_x = np.ones_like(scaled_x)  # 出力に関する勾配 (初期値は1)
        d_gamma = np.sum(d_scaled_x * normalized_x, axis=0)
        d_beta = np.sum(d_scaled_x, axis=0)
        d_normalized_x = d_scaled_x * gamma
        d_variance = np.sum(d_normalized_x * (x - mean) * (-0.5) * (variance + epsilon)**(-1.5), axis=1, keepdims=True)
        d_mean = np.sum(d_normalized_x * (-1) / np.sqrt(variance + epsilon), axis=1, keepdims=True) + \
                 np.sum(d_variance * (-2) * (x - mean) / x.shape[1], axis=1, keepdims=True)
        d_x = d_normalized_x / np.sqrt(variance + epsilon) + \
              d_variance * 2 * (x - mean) / x.shape[1] + \
              d_mean / x.shape[1]
    
        return scaled_x, {"d_x": d_x, "d_gamma": d_gamma, "d_beta": d_beta}

        # 使用例
        x = np.array([[1, 2, 3], [4, 5, 6]])
        feature_dim = x.shape[1]
        gamma = np.ones(feature_dim)
        beta = np.zeros(feature_dim)
        normalized_x, gradients = layer_normalization_with_gradients(x, gamma, beta)
        print("Normalized x:\n", normalized_x)
        print("Gradients:\n", gradients)
*/
    }

    public function __clone()
    {
        if(isset($this->gamma)) {
            $this->gamma = clone $this->gamma;
        }
        if(isset($this->beta)) {
            $this->beta = clone $this->beta;
        }
        if(isset($this->dGamma)) {
            $this->dGamma = clone $this->dGamma;
        }
        if(isset($this->dBeta)) {
            $this->dBeta = clone $this->dBeta;
        }

        $this->allocateWeights(2);
        if($this->assignedWeights) {
            $this->syncWeightVariables();
        }
    }
}
